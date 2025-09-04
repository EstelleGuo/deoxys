```markdown
### Installaion

1. **Install with Script**
    ```bash
    bash DesktopPackageInstall
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

4. **Initialize the Franka Arm with your config**
    ```bash
    ./auto_scripts/auto_arm.sh
    ```

5. **Open a Second Terminal Window**

6. **SSH into the Remote Machine**
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

1. **SSH into the Remote Machine**
    ```bash
    python Documents/WorkSpaces/nyc_ws/deoxys_control/deoxys/deoxys/sensor_interface/camera_server.py
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
