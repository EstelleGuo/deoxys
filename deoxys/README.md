```markdown
### Installation

1. **Make the installer executable**
    ```bash
    sudo chmod 777 DesktopPackageInstall
    ```
2. **Run the installer**
    ```bash
    ./DesktopPackageInstall
    ```

---

### Steps for Setting Up Franka Arm and Gripper

1. **Open a Terminal Window**

2. **SSH into the Remote Machine**
    ```bash
    ssh nuc
    ```

3. **Navigate to the Project Directory**
    ```bash
    cd Documents/WorkSpaces/nyc_ws/deoxys_control/deoxys
    ```

4. **Initialize the Franka Arm**
    ```bash
    ./auto_scripts/auto_arm.sh
    ```

5. **Open a Second Terminal Window**

6. **SSH into the Remote Machine Again**
    ```bash
    ssh nuc
    ```

7. **Navigate to the Project Directory Again**
    ```bash
    cd Documents/WorkSpaces/nyc_ws/deoxys_control/deoxys
    ```

8. **Initialize the Franka Gripper**
    ```bash
    ./auto_scripts/auto_gripper.sh
    ```

---

### Steps for Setting Up Cameras

1. **Ensure Redis Server is Running in Docker**
    ```bash
    docker run -d -p 6379:6379 redis:latest
    ```

2. **Start the Tripod Camera Node**
    - Open a new terminal window and run:
    ```bash
    cd /home/gn/Documents/Reaserach_Space/nyu_learning/deoxys_vision/scripts
    python deoxys_camera_node.py --eval --use-rgb --use-depth --visualization --camera_info tripod_cam.yml
    ```

3. **Start the Wrist Camera Node**
    - Open another new terminal window and run:
    ```bash
    cd /home/gn/Documents/Reaserach_Space/nyu_learning/deoxys_vision/scripts
    python deoxys_camera_node.py --eval --use-rgb --use-depth --visualization --camera_info wrist_cam.yml
    ```

---

### Steps for Collecting Data

1. **Plug in the Space Mouse**

2. **Navigate to the Data Collection Script**
    ```bash
    cd /home/gn/Documents/Reaserach_Space/nyu_learning/deoxys_control/deoxys/examples/demo_collection
    ```

3. **Run the Data Collection Script**
    ```bash
    python data_collection.py
    ```

---

### Steps for Creating a Dataset

1. **Navigate to the Dataset Creation Script**
    ```bash
    cd /home/gn/Documents/Reaserach_Space/nyu_learning/deoxys_control/deoxys/examples/demo_collection
    ```

2. **Run the Dataset Creation Script**
    ```bash
    python create_dataset.py
    ```
