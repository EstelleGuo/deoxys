import hid
import time


def to_int16(b1, b2):
    value = b1 | (b2 << 8)
    return value - 65536 if value >= 32768 else value


def find_spacemouse(vendor_id=9583, product_id=50741):
    print("Enumerating all HID devices:")
    devices = hid.enumerate()
    for d in devices:
        print(f"VID: {d['vendor_id']}, PID: {d['product_id']}, Product: {d['product_string']}")
        if d['vendor_id'] == vendor_id and d['product_id'] == product_id:
            print("Found SpaceMouse:", d['product_string'])
            return True
    return False


def main():
    vendor_id = 9583  # 0x256f
    product_id = 50741  # for SpaceMouse Wireless Receiver, or try 50735

    # First, let's see all available devices
    find_spacemouse(vendor_id, product_id)
    
    # Try common SpaceMouse product IDs
    common_pids = [50741]  # Different SpaceMouse models
    
    device_found = False
    working_pid = None
    
    for pid in common_pids:
        try:
            dev = hid.device()
            dev.open(vendor_id, pid)
            print(f"Successfully opened device with PID: {pid}")
            working_pid = pid
            device_found = True
            break
        except Exception as e:
            print(f"Failed to open PID {pid}: {e}")
            continue
    
    if not device_found:
        print("Could not open any SpaceMouse device!")
        return

    dev.set_nonblocking(True)
    
    print("Reading from SpaceMouse. Press Ctrl+C to quit.")
    no_data_count = 0
    
    try:
        while True:
            data = dev.read(14)  # Try reading more bytes
            if data:
                no_data_count = 0
                print(f"Raw data (len={len(data)}): {list(data)}")
                
                if len(data) >= 13 and data[0] == 1:
                    tx = to_int16(data[1], data[2])
                    ty = to_int16(data[3], data[4])
                    tz = to_int16(data[5], data[6])
                    rx = to_int16(data[7], data[8])
                    ry = to_int16(data[9], data[10])
                    rz = to_int16(data[11], data[12])
                    print(f"Translation: [{tx}, {ty}, {tz}] | Rotation: [{rx}, {ry}, {rz}]")

                elif len(data) >= 2 and data[0] == 3:
                    btn = data[1]
                    if btn == 1:
                        print("Left button pressed")
                    elif btn == 2:
                        print("Right button pressed")
                    elif btn == 0:
                        print("Button released")
                elif data[0] not in [1, 3]:
                    print(f"Unknown packet type: {data[0]}")
            else:
                no_data_count += 1
                if no_data_count % 1000 == 0:  # Print every 1000 empty reads
                    print(f"No data for {no_data_count} reads...")
            
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Exiting...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'dev' in locals():
            dev.close()


if __name__ == "__main__":
    main()
