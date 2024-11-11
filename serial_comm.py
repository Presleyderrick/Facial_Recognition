import serial
import serial.tools.list_ports

def setup_serial(baud_rate=9600):
    """Check available serial ports and connect to the first one."""
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        raise serial.SerialException("No available serial ports found.")
    
    for port in ports:
        try:
            ser = serial.Serial(port.device, baud_rate)
            print(f"Connected to serial port: {port.device}")
            return ser
        except serial.SerialException:
            continue

    raise serial.SerialException("Unable to connect to any serial port.")


def send_serial_data(ser, data):
    """Send data through the serial port."""
    if ser and ser.is_open:
        ser.write(data.encode())
        print(f"Sent data: {data}")
