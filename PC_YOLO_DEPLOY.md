# PC YOLO inspection deployment

## 1. PC network

Set the PC Ethernet IP to the same network segment as the Raspberry Pi.

Example:

- PC: `172.16.68.100`
- Raspberry Pi: `172.16.68.111`

The Raspberry Pi web page address is:

```text
http://172.16.68.111:8080
```

The Raspberry Pi backend API address used by the web page is:

```text
http://172.16.68.111:7777
```

The PC YOLO service address is:

```text
http://172.16.68.100:8080
```

Allow TCP port `8080` through the PC firewall.

## 2. Install PC dependencies

Install Python on the PC, then run:

```bat
cd /d C:\Users\16600\Desktop\chanxian
python -m pip install -r requirements_pc_yolo.txt
```

## 3. Configure measurement calibration

Edit `pc_yolo_config.json`.

If the camera is fixed, use a known-size aluminum sheet to calculate:

```text
mm_per_pixel_x = real_width_mm / measured_width_px
mm_per_pixel_y = real_height_mm / measured_height_px
```

Then fill:

```json
{
  "mm_per_pixel_x": 0.1,
  "mm_per_pixel_y": 0.1
}
```

If these values are `null`, the service still returns and displays pixel sizes, but millimeter sizes will be empty.

## 4. Start PC YOLO service

```bat
cd /d C:\Users\16600\Desktop\chanxian
start_pc_yolo_service.bat
```

Check:

```text
http://127.0.0.1:8080/health
```

## 5. Configure Raspberry Pi backend

Edit `GrabImage/config.ini`:

```ini
[Inference]
predict_url = http://172.16.68.100:8080/predict
timeout = 5
```

Replace `172.16.68.100` with the PC Ethernet IP if your PC IP changes.

Restart the Raspberry Pi backend after changing this value.

## 6. Result flow

The Raspberry Pi backend sends the captured aluminum image to the PC `/predict` endpoint.

The PC returns:

- aluminum bounding box and size
- defect boxes and sizes
- existing-compatible defect result fields

The Raspberry Pi backend writes:

- `GrabImage/detect_files/detect.jpg`
- `GrabImage/detect_files/detect.json`

The web page shows the annotated result image. The `/get_detect_pic` API also returns the structured measurement data in the `measurement` field.
