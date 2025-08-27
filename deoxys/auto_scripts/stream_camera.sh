# Check if port 10001 is in use
if ss -ltn | grep -q ':10001 '; then
	echo "camera server is streaming !!"
else
	echo "Starting camera server on port 10001..."
	# Activate conda environment and run camera_server.py
	source ~/anaconda3/etc/profile.d/conda.sh
	conda activate nyu_ws
	python ../deoxys/sensor_interface/camera_server.py
fi