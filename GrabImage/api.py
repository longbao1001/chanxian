from concurrent.futures import ThreadPoolExecutor
#from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as compare_ssim
import sys
import time
import uuid
import requests
import numpy as np
import threading
import os
import cv2
from ctypes import *
from flask import Flask, render_template, Response, request
from PIL import Image, ImageFont, ImageDraw
import json
import base64
import configparser
from queue import Queue
import datetime
import serial
import database
from flask_cors import CORS
from logger import setup_log

logger = setup_log('detect')
# 缺陷对应关系，将拼音转化为中文
base_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(base_dir, 'config.ini')
cf = configparser.ConfigParser()
cf.read(config_path, encoding='utf-8-sig')
defect_name = dict(cf.items('defect_name'))
predict_url = cf.get('Inference', 'predict_url', fallback='http://172.16.68.110:8080/predict')
predict_timeout = cf.getfloat('Inference', 'timeout', fallback=5.0)

# 报警器串口指令
#ser = serial.Serial("/dev/BAOJING", 9600, 8, stopbits=1)
bao_jing = '7E FF 06 03 00 00 01 EF'
baojing = bytes.fromhex(bao_jing)

sys.path.append("../MvImport")
from MvCameraControl_class import *

g_bExit = False
app = Flask(__name__)
CORS(app, supports_credentials=True)
# 保留缺陷图片数量
save_image_num = 10
# 缺陷图片路径
defect_image_file = os.path.join(base_dir, "files") + os.sep
# 当前原始图片路径
original_image_path = os.path.join(base_dir, "original_files", "original.jpg")
# 当前缺陷图片存放路径
detect_image_path = os.path.join(base_dir, "detect_files", "detect.jpg")
detect_result_path = os.path.join(base_dir, "detect_files", "detect.json")
measure_history_dir = os.path.join(base_dir, "measure_history")

# 创建存放缺陷检测图片目录
if(os.path.exists(os.path.join(base_dir, "files"))==False):
    os.mkdir(os.path.join(base_dir, "files"))

# 创建存放当前缺陷检测图片存放目录
if(os.path.exists(os.path.join(base_dir, "detect_files"))==False):
    os.mkdir(os.path.join(base_dir, "detect_files"))

# 创建存放当前原始图片存放目录
if(os.path.exists(os.path.join(base_dir, "original_files"))==False):
    os.mkdir(os.path.join(base_dir, "original_files"))

# 创建尺寸历史记录目录
if(os.path.exists(measure_history_dir)==False):
    os.mkdir(measure_history_dir)

executor = ThreadPoolExecutor()
arm_action_lock = threading.Lock()
last_arm_action_time = 0

camstream_cap = []

# 图片处理队列，先进先出
q1 = Queue(maxsize=0)
# 初始化图片
stream_image = np.zeros((512, 512, 3), dtype=np.uint8)

# 置信度修改
def post_processing(out):
    # 创建 'result' 列表的副本
    new_result = out['result'][:]

    # 根据条件过滤项目
    new_result = [item for item in new_result if (item['score'] >= 0.7 and item['class_name'] != 'zhen_kong') or (item['class_name'] == 'zhen_kong' and item['score'] >= 0.2)]

    # 更新输出字典中的 'result' 和 'len' 键
    out['result'] = new_result
    out['len'] = len(new_result)
    return out


#报警
def save_detect_result(uid, out_put):
    result = dict(out_put)
    result['uuid'] = uid
    result['created_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(detect_result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)
    history_name = '{}_{}.json'.format(result['created_time'].replace('-', '').replace(':', '').replace(' ', '_'), uid)
    history_path = os.path.join(measure_history_dir, history_name)
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)


def format_size(size):
    if not size:
        return ''
    width = size.get('width_mm')
    height = size.get('height_mm')
    unit = 'mm'
    if width is None or height is None:
        width = size.get('width_px')
        height = size.get('height_px')
        unit = 'px'
    if width is None or height is None:
        return ''
    return '{:.2f}x{:.2f}{}'.format(float(width), float(height), unit)


def ensure_bgr_image(image):
    try:
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    except Exception:
        pass
    return image


def draw_measurements(image, out_put, predict_time):
    image = ensure_bgr_image(image)
    detectfps = "(Capture) {:.1f} FPS".format(predict_time)
    cv2.putText(image, detectfps, (180, 30), cv2.FONT_HERSHEY_COMPLEX, 0.3, (38, 0, 255), 1)

    aluminum = out_put.get('aluminum') or {}
    aluminum_text = format_size(aluminum.get('size') or aluminum)
    if aluminum_text:
        cv2.putText(image, 'Al: {}'.format(aluminum_text), (10, 18), cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 180, 255), 1)

    for item in out_put.get('result', []):
        loc = item.get('loc')
        if not loc or item.get('class_name') == 'zheng_chang':
            continue
        x1, y1, x2, y2 = [int(v) for v in loc]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
        size_text = format_size(item.get('size') or item)
        label = '{} {:.2f}%'.format(defect_name.get(item.get('class_name'), item.get('class_name')), item.get('score', 0) * 100)
        if size_text:
            label = '{} {}'.format(label, size_text)
        try:
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            font = ImageFont.truetype("Font/platech.ttf", 9, encoding="utf-8")
            draw = ImageDraw.Draw(img)
            draw.text((max(0, x1 - 10), max(0, y1 - 12)), label, font=font, fill="green")
            image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception:
            cv2.putText(image, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_COMPLEX, 0.35, (0, 180, 0), 1)
    return image


def detect_summary(out_put):
    aluminum = out_put.get('aluminum') or {}
    aluminum_text = format_size(aluminum.get('size') or aluminum)
    defect_sizes = []
    for item in out_put.get('result', []):
        if item.get('class_name') == 'zheng_chang':
            continue
        size_text = format_size(item.get('size') or item)
        if size_text:
            defect_sizes.append('{} {}'.format(defect_name.get(item.get('class_name'), item.get('class_name')), size_text))
    parts = []
    if aluminum_text:
        parts.append('Al {}'.format(aluminum_text))
    parts.extend(defect_sizes[:3])
    return '; '.join(parts)


def json_response(data, status=200):
    return Response(json.dumps(data, ensure_ascii=False), status=status, mimetype='application/json')


def load_mimo_config():
    parser = configparser.ConfigParser()
    parser.read(config_path, encoding='utf-8-sig')
    api_key = os.environ.get('MIMO_API_KEY') or parser.get('Mimo', 'api_key', fallback='')
    return {
        'api_key': api_key.strip(),
        'proxy_url': parser.get('Mimo', 'proxy_url', fallback='').strip(),
        'base_url': parser.get('Mimo', 'base_url', fallback='https://api.xiaomimimo.com/v1/chat/completions').strip(),
        'model': parser.get('Mimo', 'model', fallback='mimo-v2.5').strip(),
        'timeout': parser.getfloat('Mimo', 'timeout', fallback=30.0),
        'max_completion_tokens': parser.getint('Mimo', 'max_completion_tokens', fallback=1024),
    }


def image_to_data_url(path):
    with open(path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
    return 'data:image/jpeg;base64,' + encoded


def latest_measurement():
    if not os.path.exists(detect_result_path):
        return {}
    with open(detect_result_path, 'r', encoding='utf-8') as f:
        measurement = json.load(f)
    measurement['summary'] = detect_summary(measurement)
    return measurement


def build_mimo_prompt(measurement):
    measurement_text = json.dumps(measurement, ensure_ascii=False)
    if len(measurement_text) > 4000:
        measurement_text = measurement_text[:4000] + '...'
    return (
        '你是工业铝片缺陷质检智能体。请结合图片和尺寸数据分析当前铝片缺陷。'
        '请用中文输出，内容包括：1. 铝片整体情况；2. 缺陷类型；3. 缺陷尺寸和面积解读；'
        '4. 是否建议判为NG；5. 给现场操作员的处理建议。'
        '不要编造没有看到的数据；如果尺寸数据缺失，请明确说明。\n\n'
        '当前尺寸数据如下：\n{}'.format(measurement_text)
    )


def bao_jing_async():
    try:
        global ser
        ser.write(baojing)
    except Exception as e:
        logger.error('bao jing error：{}'.format(str(e)))
        ser = serial.Serial("/dev/BAOJING", 9600, 8, stopbits=1)
        time.sleep(0.1)
        ser.write(baojing)
        logger.info("reboot init baojing")

def initMvCamera():
    SDKVersion = MvCamera.MV_CC_GetSDKVersion()
    logger.info("SDKVersion[0x%x]" % SDKVersion)

    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

    # ch:枚举设备 | en:Enum device
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0:
        logger.info("enum devices fail! ret[0x%x]" % ret)
        sys.exit()

    if deviceList.nDeviceNum == 0:
        logger.info("find no device!")
        sys.exit()

    logger.info("Find %d devices!" % deviceList.nDeviceNum)

    for i in range(0, deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            logger.info("\ngige device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                strModeName = strModeName + chr(per)
            logger.info("device model name: %s" % strModeName)

            nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
            nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
            nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
            nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
            logger.info("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            logger.info("\nu3v device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            logger.info("device model name: %s" % strModeName)

            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            logger.info("user serial number: %s" % strSerialNumber)

    nConnectionNum = 0

    if int(nConnectionNum) >= deviceList.nDeviceNum:
        logger.info("intput error!")
        sys.exit()

    # ch:创建相机实例 | en:Creat Camera Object
    cam = MvCamera()
    # ch:选择设备并创建句柄| en:Select device and create handle
    stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

    ret = cam.MV_CC_CreateHandle(stDeviceList)
    #cam.MV_CC_CreateHandle(stDeviceList)
    #if ret != 0:
    #    logger.info ("create handle fail! ret[0x%x]" % ret)
    #    sys.exit()

    # ch:打开设备 | en:Open device
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    #cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    #if ret != 0:
    #    logger.info ("open device fail! ret[0x%x]" % ret)
    #    sys.exit()

    # ch:设置触发模式为off | en:Set trigger mode as off
    ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    #cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    #if ret != 0:
    #    logger.info ("set trigger mode fail! ret[0x%x]" % ret)
    #    sys.exit()

    # ch:获取数据包大小 | en:Get payload size
    stParam =  MVCC_INTVALUE()
    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

    ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
    #cam.MV_CC_GetIntValue("PayloadSize", stParam)
    #if ret != 0:
    #    logger.info ("get payload size fail! ret[0x%x]" % ret)
    #    sys.exit()
    nPayloadSize = stParam.nCurValue

    # ch:开始取流 | en:Start grab image
    ret = cam.MV_CC_StartGrabbing()
    #cam.MV_CC_StartGrabbing()
    #if ret != 0:
    #    logger.info ("start grabbing fail! ret[0x%x]" % ret)
    #    sys.exit()

    data_buf = (c_ubyte * nPayloadSize)()

    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    return cam, data_buf, nPayloadSize, stFrameInfo

# 获取7天数据
def day_get(d):
    for i in range(0,7):
        oneday = datetime.timedelta(days=i)
        day = d - oneday
        date_to = datetime.datetime(day.year, day.month, day.day)
        yield str(date_to)[5:10]

def get_day():
    d = datetime.datetime.now()
    qq =day_get(d)
    list =[]
    for obj in qq:
        list.append(obj)
    list_week_day = list[::-1]
    return list_week_day


# 抓取缺陷铝片
def zhuaqu(flags):
    global last_arm_action_time
    interval = cf.getfloat("Configuration", "arm_action_interval", fallback=5.0)
    now = time.time()
    if now - last_arm_action_time < interval:
        logger.info('skip arm action {}, cooldown {:.1f}s'.format(flags, interval))
        return
    if not arm_action_lock.acquire(False):
        logger.info('skip arm action {}, previous action is running'.format(flags))
        return
    try:
        speed = cf.get("Configuration", "speed")
        time1 = cf.get("Configuration", "time")
        url1 = 'http://172.16.68.111:8899/grab'
        data = {"flags": flags, "speed": speed, "time": time1}
        requests.post(url1, json=data, timeout=5)
        logger.info('arm action sent：{}'.format(flags))
    except Exception as e:
        logger.error('arm action error：{}'.format(str(e)))
    finally:
        last_arm_action_time = time.time()
        arm_action_lock.release()


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST')  # Put any other methods you need here
    return response


# 获取抓取配置文件信息
@app.route('/get_conf', methods=['GET'])
def get_conf():
    jj = []
    j = dict(cf.items('Configuration'))
    j['defect_name'] = defect_name
    jj.append(j)
    return json.dumps(jj)


# 修改抓取配置文件信息
@app.route('/change_conf', methods=['POST'])
def change_conf():
    data = json.loads(request.get_data(as_text=True))
    time1 = data['time']
    if len(time1)>0:
        cf.set("Configuration", "time", time1)
    speed = data['speed']
    if len(speed)>0:
        cf.set("Configuration", "speed", speed)
    grab_position = data['grab_position']
    if len(grab_position)>0:
        cf.set("Configuration", "grab_position", grab_position)
    release_position = data['release_position']
    if len(release_position)>0:
        cf.set("Configuration", "release_position", release_position)
    # 写入配置文件
    with open(config_path, 'w', encoding='utf-8') as f:
        cf.write(f)
    return data


# 获取最新4张历史检测图片
@app.route('/get_history', methods=['GET'])
def get_history():
    sql_1 = database.select_instructions('*', 'defect_list', 'where path is not null')
    sql_2 = database.select_instructions('*', '('+sql_1+')', "where path is not 'detect.jpg' order by id DESC")
    sql_3 = database.select_instructions('distinct path', '('+sql_2+')', 'limit 4')
    values1 = database.select_data(sql_3)
    image_files = []
    for i in range (0, len(values1)):
        image_file = values1[i][0]
        with open(image_file, 'rb') as f:
            image = f.read()
            image_base64 = "data:image/jpg;base64," + str(base64.b64encode(image), encoding='utf-8')
            sql_4 = database.select_instructions('name', 'defect_list', "where path='{}'".format(image_file))
            values2 = database.select_data(sql_4)
            name = values2
            j = {"name": name, "img": image_base64}
            image_files.append(j)
    return json.dumps(image_files)


# 获取检测原始图片
@app.route('/get_original_pic', methods=['GET'])
def get_original_pic():
    image_files = []
    if os.path.exists(original_image_path):
        with open(original_image_path, 'rb') as f:
            image = f.read()
            image_base64 = "data:image/jpg;base64," + str(base64.b64encode(image), encoding='utf-8')
            image_files.append(image_base64)
    return json.dumps(image_files)


# 获取检测后带有缺陷的图片
@app.route('/get_detect_pic', methods=['GET'])
def get_detect_pic():
    image_files = []
    if os.path.exists(detect_image_path):
        sql_1 = database.select_instructions('*', 'defect_list', "where path='detect.jpg' order by id DESC")
        sql_2 = database.select_instructions('distinct uuid', '('+sql_1+')', 'limit 1')
        values1 = database.select_data(sql_2)
        name = []
        if values1:
            uuid = values1[0][0]
            sql_3 = database.select_instructions('name', 'defect_list', "where uuid='{}' and path='detect.jpg'".format(uuid))
            name = database.select_data(sql_3)
        measurement = {}
        if os.path.exists(detect_result_path):
            with open(detect_result_path, 'r', encoding='utf-8') as f:
                measurement = json.load(f)
            measurement['summary'] = detect_summary(measurement)
        with open(detect_image_path, 'rb') as f:
            image = f.read()
            image_base64 = "data:image/jpg;base64," + str(base64.b64encode(image), encoding='utf-8')
            j = {"name": name, "img": image_base64, "measurement": measurement}
            image_files.append(j)
    return json.dumps(image_files)


@app.route('/get_measure_history', methods=['GET'])
def get_measure_history():
    limit = request.args.get('limit', default=20, type=int)
    limit = max(1, min(limit, 100))
    records = []
    if not os.path.exists(measure_history_dir):
        return json.dumps(records, ensure_ascii=False)

    files = []
    for name in os.listdir(measure_history_dir):
        if name.endswith('.json'):
            files.append(os.path.join(measure_history_dir, name))
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)

    for path in files[:limit]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                record = json.load(f)
            record['summary'] = detect_summary(record)
            records.append(record)
        except Exception as e:
            logger.error('read measure history error：{}'.format(str(e)))
    if not records and os.path.exists(detect_result_path):
        try:
            with open(detect_result_path, 'r', encoding='utf-8') as f:
                record = json.load(f)
            record['summary'] = detect_summary(record)
            records.append(record)
        except Exception as e:
            logger.error('read current measure as history error：{}'.format(str(e)))
    return json.dumps(records, ensure_ascii=False)


@app.route('/mimo_analyze_defect', methods=['POST', 'GET'])
def mimo_analyze_defect():
    mimo_config = load_mimo_config()
    image_path = detect_image_path if os.path.exists(detect_image_path) else original_image_path
    if not os.path.exists(image_path):
        return json_response({
            'status': 'no_image',
            'message': '没有找到可分析的检测图片，请先完成一次识别。'
        }, status=400)

    measurement = latest_measurement()
    if mimo_config['proxy_url']:
        try:
            with open(image_path, 'rb') as f:
                files = {'image_file': ('detect.jpg', f, 'image/jpeg')}
                data = {'measurement': json.dumps(measurement, ensure_ascii=False)}
                response = requests.post(mimo_config['proxy_url'], files=files, data=data, timeout=(3, mimo_config['timeout']))
            return Response(response.content, status=response.status_code, mimetype='application/json')
        except Exception as e:
            logger.error('mimo proxy analyze error：{}'.format(str(e)))
            return json_response({
                'status': 'error',
                'message': 'PC MiMo 代理调用失败：{}'.format(str(e))
            }, status=500)

    if not mimo_config['api_key']:
        return json_response({
            'status': 'missing_key',
            'message': 'MiMo API Key 未配置。请在 GrabImage/config.ini 的 [Mimo] api_key 中填写 key，或设置环境变量 MIMO_API_KEY。'
        }, status=400)

    payload = {
        'model': mimo_config['model'],
        'messages': [
            {
                'role': 'system',
                'content': '你是专业的工业视觉质检助手，当前日期是2026年06月04日。'
            },
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': image_to_data_url(image_path)
                        }
                    },
                    {
                        'type': 'text',
                        'text': build_mimo_prompt(measurement)
                    }
                ]
            }
        ],
        'max_completion_tokens': mimo_config['max_completion_tokens']
    }

    try:
        response = requests.post(
            mimo_config['base_url'],
            headers={
                'api-key': mimo_config['api_key'],
                'Content-Type': 'application/json'
            },
            json=payload,
            timeout=mimo_config['timeout']
        )
        response.raise_for_status()
        result = response.json()
        message = result.get('choices', [{}])[0].get('message', {})
        return json_response({
            'status': 'ok',
            'model': mimo_config['model'],
            'analysis': message.get('content', ''),
            'measurement_summary': measurement.get('summary', ''),
            'created_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        logger.error('mimo analyze error：{}'.format(str(e)))
        return json_response({
            'status': 'error',
            'message': 'MiMo 分析失败：{}'.format(str(e))
        }, status=500)


# 获取检测数
@app.route('/get_num', methods=['GET'])
def get_num():
    sql_1 = database.select_instructions('*', 'defect_list', 'where path is not null')
    sql_2 = database.select_instructions('*', '('+sql_1+')', "where path is not 'detect.jpg'")
    sql_3 = database.select_instructions('name, count(1) AS counts', '('+sql_2+')', 'group by name')
    values1 = database.select_data(sql_3)
    other_num = values1
    json1 = []
    json2 = []
    for i in range (0, len(other_num)):
        j = {other_num[i][0]: other_num[i][1]}
        json2.append(j)
    json1.append(json2)
    return json.dumps(json1)


# 获取30天内检测信息
@app.route('/get_this_month_num', methods=['GET'])
def get_this_month_num():
    sql_1 = database.select_instructions('*', 'defect_list', 'where path is not null')
    sql_2 = database.select_instructions('*', '('+sql_1+')', "where path is not 'detect.jpg'")
    sql_3 = database.select_instructions('*', '('+sql_2+')', "WHERE DATE(CreatedTime) >= DATE('now', 'start of month', '+1 seconds')")
    sql_4 = database.select_instructions('name, count(1) AS counts', '('+sql_3+')', 'group by name')
    values1 = database.select_data(sql_4)
    other_num = values1
    json1 = []
    json2 = []
    for i in range (0, len(other_num)):
        j = {other_num[i][0]: other_num[i][1]}
        json2.append(j)
    json1.append(json2)
    return json.dumps(json1)


# 获取一周内的检测信息
@app.route('/get_seven_days_num', methods=['GET'])
def get_seven_days_num():
    today_num = database.select_day_data("+0", "+1")
    yesterday_num = database.select_day_data("-1", "+0")
    two_days_ago_num = database.select_day_data("-2", "-1")
    three_days_ago_num = database.select_day_data("-3", "-2")
    four_days_ago_num = database.select_day_data("-4", "-3")
    five_days_ago_num = database.select_day_data("-5", "-4")
    six_days_ago_num = database.select_day_data("-6", "-5")
    list_week_day = get_day()
    j = {list_week_day[6]: today_num, list_week_day[5]: yesterday_num, list_week_day[4]: two_days_ago_num, list_week_day[3]: three_days_ago_num, list_week_day[2]: four_days_ago_num, list_week_day[1]: five_days_ago_num, list_week_day[0]: six_days_ago_num}
    return json.dumps(j)


# 获取统计信息
@app.route('/get_statistics', methods=['GET'])
def get_statistics():
    sql_1 = database.select_instructions('*', 'defect_list', 'where prediction_time is not null')
    sql_2 = database.select_instructions('sum(prediction_time)', '('+sql_1+')', '')
    values1 = database.select_data(sql_2)
    total_prediction_time = values1[0][0]
    sql_3 = database.select_instructions('*', 'defect_list', 'where score is not null')
    sql_4 = database.select_instructions('sum(score)', '('+sql_3+')', '')
    values2 = database.select_data(sql_4)
    total_score = values2[0][0]
    sql_5 = database.select_instructions('count()', '('+sql_1+')', '')
    values3 = database.select_data(sql_5)
    num = values3[0][0]
    average_prediction_time = 0
    average_score = 0
    if  num:
        average_prediction_time = total_prediction_time/num
        average_score = (total_score/num)*100
    sql_6 = database.select_instructions('count(distinct uuid)', 'defect_list', '')
    values4 = database.select_data(sql_6)
    total_num = values4[0][0]
    sql_7 = database.select_instructions('count(distinct uuid)', 'defect_list', "where name='hua_shang' or name='zang_wu' or name='zhen_kong'")
    values5 = database.select_data(sql_7)
    defect_num = values5[0][0]
    j = {"average_score": average_score, "average_prediction_time": average_prediction_time, "total_num": total_num, "defect_num": defect_num}
    return json.dumps(j)


# 将视频流图片和原始图片进行比较，获取对应的差异值。值为1表示两张图片一样，值越小，差异越大。
def compare_image(image):
    # 传入图片路径，读取图片
    image_a = cv2.imread("yuanshi.jpg")
    image_a = cv2.resize(image_a, (64, 48), interpolation=cv2.INTER_AREA)
    #image_b = cv2.imread(opt.image)
    # 使用色彩空间转化函数 cv2.cvtColor( )进行色彩空间的转换
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    #gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    # 计算图像相似度并圈出不同处
    t0 = time.time()
    (score, diff) = compare_ssim(gray_a, image, full=True)
    t1 = time.time()
    #logger.info("Compare_time is {}ms ".format((t1-t0)*1000))
    logger.info("SSIM: {}".format(score))
    return score

class Producer(threading.Thread):
    def run(self):
        cam, data_buf, nPayloadSize, stFrameInfo = initMvCamera()
        while True:
            ret = cam.MV_CC_GetOneFrameTimeout(data_buf, nPayloadSize, stFrameInfo, 10000)
            if ret != 0:
                logger.info('海康相机状态：{}'.format(ret))
            if ret == 2147483655:
                logger.info("相机重启")
                ret = cam.MV_CC_StopGrabbing()
                #ch:关闭设备 | Close device
                ret = cam.MV_CC_CloseDevice()
                # ch:销毁句柄 | Destroy handle
                ret = cam.MV_CC_DestroyHandle()
                cam, data_buf, nPayloadSize, stFrameInfo = initMvCamera()
                continue
            time.sleep(0.01)
            if ret == 0:
                image = np.asarray(data_buf).reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
                #3072*2048 768 * 512
                image = cv2.resize(image, (int(stFrameInfo.nWidth/4), int(stFrameInfo.nHeight/4)), interpolation=cv2.INTER_AREA)
                #图片处理队列
                q1.put(image)
                #视频流队列
                global stream_image
                stream_image = image


class Consumer(threading.Thread):
    def run(self):
        k = 0
        count = 3
        diff1, diff2, diff3 = 0, 0, 0
        arr = [0] * 9
        arr_img = [0, 0]
        database.create_database()
        while True:
            time.sleep(0.01)
            if not q1.empty():
                try:
                    t1 = time.time()
                    # 每3帧处理一次，提高抓拍到铝片的概率。
                    if k == 0:
                        image = q1.get()
                        # 保存最近的两张图片,当触发检测之后,获取前一张图片
                        arr_img[0] = arr_img[1]
                        arr_img[1] = image
                        # 图片处理，高斯滤波、膨胀、二值化，用于得到只有黑白像素的图片，用于判断是否有铝片进入视野
                        black_image = cv2.resize(image, (64, 48), interpolation=cv2.INTER_AREA)
                        blurred = cv2.GaussianBlur(black_image, (21, 21), 0)
                        _,img1=cv2.threshold(blurred,100,255,cv2.THRESH_BINARY)
                        thresh = cv2.dilate(img1, None, iterations=4)
                        _,img2=cv2.threshold(thresh,0.3,255,cv2.THRESH_BINARY)
                        # 获取白色像素的占比
                        wt = np.count_nonzero(img2)
                        x, y = img2.shape
                        rate1 = wt / (x * y)
                        diff_new = compare_image(black_image)
                        arr[:-1] = arr[1:]
                        arr[-1] = diff_new
                        # 计算标准差
                        arr_std = np.std(arr)
                        # 计算均值
                        arr_mean = np.mean(arr)
                        # 判断视野是否长期处于稳定状态
                        if arr_std < 0.01 and arr_mean < 0.8 and all(arr):
                            #yuanshi_img = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
                            cv2.imwrite("yuanshi.jpg", image)
                        # 获取当前帧和前一帧的差异值大小
                        s1 = diff_new - diff3
                        # 获取前一帧和前两帧的差异值大小
                        s2 = diff3 - diff2
                        # 获取前两帧和前三帧的差异值大小
                        s3 = diff2 - diff1
                        if diff_new < 0.9:
                            # 判断铝片进入视野后的差异值大小，其中铝片全部进入视野差异值最小，判断最小峰值处用于推理
                            logger.info('diff_new:{},diff3:{},diff2:{},diff1:{},rate1:{}'.format(str(diff_new),str(diff3),str(diff2),str(diff1),str(rate1)))
                            if s1 > 0 and s2 < 0 and s3 < 0 and rate1 > 0.1:
                                uid = str(uuid.uuid1())
                                file_name = os.path.join(defect_image_file, '{}.jpg'.format(uid))
                                image2 = arr_img[0]
                                original_image = image2.copy()
                                bs = cv2.imencode(".jpg", image2)[1].tobytes()
                                url = predict_url
                                files = { 'image_file': bs }
                                start_time = time.time()
                                logger.info('开始检测')
                                f = requests.post(url, files=files, timeout=predict_timeout)
                                logger.info('推理服务耗费：{}'.format(time.time() - start_time))
                                predict_time = 10/(time.time() - t1)
                                out_put = json.loads(f.text)
                                logger.info(out_put)
                                out_put = post_processing(out_put)
                                save_detect_result(uid, out_put)
                                logger.info("处理后")
                                logger.info(out_put)
                                if out_put['len'] > 0:
                                    only_zheng_chang = all(out_put_item['class_name'] == 'zheng_chang' for out_put_item in out_put['result'])
                                    if only_zheng_chang:
                                        executor.submit(zhuaqu, "OK")
                                        image2 = draw_measurements(image2, out_put, predict_time)
                                        cv2.imwrite(file_name, image2)
                                        database.insert_data(uid, file_name, 'zheng_chang', None, None)
                                        database.insert_data(uid, 'detect.jpg', 'zheng_chang', None, None)
                                        cv2.imwrite(original_image_path, original_image)
                                        cv2.imwrite(detect_image_path, image2)
                                    else:
                                        logger.info("准备报警")
                                        executor.submit(bao_jing_async)
                                        logger.info("报警完成")
                                        time.sleep(0.1)
                                        executor.submit(zhuaqu, "NG")
                                        logger.info("检测到缺陷，触发报警和抓取动作")
                                        image2 = ensure_bgr_image(image2)
                                        detectfps = "(Capture) {:.1f} FPS".format(predict_time)
                                        cv2.putText(image2, detectfps, (180, 30), cv2.FONT_HERSHEY_COMPLEX, 0.3, (38,0,255), 1)
                                        aluminum = out_put.get('aluminum') or {}
                                        aluminum_text = format_size(aluminum.get('size') or aluminum)
                                        if aluminum_text:
                                            cv2.putText(image2, 'Al: {}'.format(aluminum_text), (10, 18), cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 180, 255), 1)
                                        for out_put_item in out_put['result']:
                                            class_name = out_put_item['class_name']
                                            if class_name != 'zheng_chang':
                                                class_names = defect_name.get(class_name, class_name)
                                                score = out_put_item['score']
                                                loc = out_put_item['loc']
                                                prediction_time = out_put_item['prediction_time']
                                                x1 = int(loc[0])
                                                y1 = int(loc[1])
                                                x2 = int(loc[2])
                                                y2 = int(loc[3])
                                                cv2.rectangle(image2, (x1,y1), (x2, y2), (255,0,0), 1)
                                                img = Image.fromarray(cv2.cvtColor(image2,cv2.COLOR_BGR2RGB))
                                                font = ImageFont.truetype("Font/platech.ttf", 9, encoding="utf-8")
                                                draw = ImageDraw.Draw(img)
                                                size_text = format_size(out_put_item.get('size') or out_put_item)
                                                label = '{} {:.2f}%'.format(class_names, score*100)
                                                if size_text:
                                                    label = '{} {}'.format(label, size_text)
                                                draw.text((max(0, x1-10), max(0, y1-10)), label, font=font, fill="green")
                                                image2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                                                cv2.imwrite(file_name, image2)
                                                # 缺陷信息插入数据库
                                                database.insert_data(uid, file_name, class_name, prediction_time, score)
                                                database.insert_data(uid, 'detect.jpg', class_name, None, None)
                                                logger.info("数据库写入完成")
                                                cv2.imwrite(original_image_path, original_image)
                                                cv2.imwrite(detect_image_path, image2)
                                else:
                                    executor.submit(zhuaqu, "OK")
                                    image2 = draw_measurements(image2, out_put, predict_time)
                                    cv2.imwrite(file_name, image2)
                                    database.insert_data(uid, file_name, 'zheng_chang', None, None)
                                    database.insert_data(uid, 'detect.jpg', 'zheng_chang', None, None)
                                    cv2.imwrite(original_image_path, original_image)
                                    cv2.imwrite(detect_image_path, image2)

                        diff1 = diff2
                        diff2 = diff3
                        diff3 = diff_new
                    else:
                        image = q1.get()

                    k = k + 1
                    k = k % count
                except Exception as e:
                    logger.error('consumer thread error：{}'.format(str(e)))


def get_frame():
    frame = None
    try:
        # 因为opencv读取的图片并非jpeg格式，因此要用motion JPEG模式需要先将图片转码成jpg格式图片
        ret, jpeg = cv2.imencode('.jpg', stream_image)
        frame = jpeg.tobytes()
    except Exception as e:
        logger.error('get frame error：%s', e, exc_info=True)
    return frame


def gen():
    while True:
        time.sleep(0.1)
        frame = get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/img')  # 这个地址返回视频流响应
def video_feed():
    if not camstream_cap:
        p = Producer()
        camstream_cap.append(p)
        p.start()
        c = Consumer()
        c.start()
    q1.queue.clear()
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace;boundary=frame')


@app.route('/')  # 主页
def index():
    # jinja2模板，具体格式保存在index.html文件中
    return render_template('index.html')


class DeleteImg(object):
    def __init__(self):
        self.__delete_thread = threading.Thread(target=self._delete)
        self.__delete_thread.start()

    def _delete(self):
        sql_1 = database.select_instructions('distinct path', 'defect_list', "where path is not null and path != 'detect.jpg' order by id DESC limit 10")
        values1 = database.select_data(sql_1)
        image_files = []
        for i in range (0, len(values1)):
            image_file = values1[i][0]
            image_files.append(image_file)
        file_list = os.listdir(defect_image_file)
        for f in file_list:
            if (defect_image_file + f) not in image_files:
                os.remove(defect_image_file + f)

delete_img = DeleteImg()
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, use_reloader=False, port=7777)
