# import pygame

# pygame.init()

# # Initialize the joystick (gamepad)
# joystick = pygame.joystick.Joystick(0)  # 0 for the first gamepad
# joystick.init()

# print("Gamepad initialized")
# print(f"Number of buttons: {joystick.get_numbuttons()}")

# try:
#     while True:
#         pygame.event.pump()  # Update the state of the gamepad
        
#         for i in range(joystick.get_numbuttons()):
#             if joystick.get_button(i):  # If button i is pressed
#                 print(f"Button {i} is pressed")

#         pygame.time.wait(100)  # Wait 100ms to reduce CPU load
# except KeyboardInterrupt:
#     print("Exiting...")
#     pygame.quit()

from deoxys.deoxys.utils.io_devices.gamepad_old import ZikwayGamepad
import time

gamepad = ZikwayGamepad()
gamepad.start_control()

try:
    for i in range(1000):
        print(gamepad.control, gamepad.control_gripper)
        print(gamepad._collecting)
        time.sleep(0.02)
except KeyboardInterrupt:
    print("Interrupted by user")
finally:
    gamepad.close()