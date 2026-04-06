import depthai as dai

print("=== OAK Camera Detection Test ===\n")

devices = dai.Device.getAllAvailableDevices()

print(f"Found {len(devices)} OAK device(s):")

for d in devices:
    try:
        mxid = d.getMxId()
        state = d.state
        usb_speed = getattr(d, 'getUsbSpeed', lambda: "N/A")()
        print(f" → MxID: {mxid} | State: {state} | USB Speed: {usb_speed}")
    except Exception as e:
        print(f" → Device found but error reading info: {e}")

if not devices:
    print("\n❌ No OAK device found!")
    print("\nPossible reasons and fixes:")
    print("1. Camera is not plugged in")
    print("2. Try a different USB port (preferably blue USB 3.0 port)")
    print("3. Unplug the OAK, wait 10 seconds, plug it back in")
    print("4. Try a different USB cable if possible")
    print("5. Make sure no other program is using the camera")
else:
    print("\n✅ Success! OAK device is detected by your computer.")
    print("The hardware is working. The problem is likely in the pipeline code.")