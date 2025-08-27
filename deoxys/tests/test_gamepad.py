# pip install evdev
from evdev import InputDevice, categorize, ecodes, list_devices

for dev_path in list_devices():
    dev = InputDevice(dev_path)
    print(dev_path, dev.name)

# Use the actual device node for your gamepad
gamepad = InputDevice(dev_path)

print("Connected:", gamepad.name)

for event in gamepad.read_loop():
    
    if event.type == ecodes.EV_KEY:
        print('Button', ecodes.KEY.get(event.code, event.code),
              'pressed' if event.value else 'released')
        
    elif event.type == ecodes.EV_ABS:
        abscode = ecodes.ABS.get(event.code, event.code)
        print('Axis', abscode, 'value', event.value)
