#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Bool
from sensor_msgs.msg import Joy
import serial
from threading import Lock
import time

class SerialController(Node):
    def __init__(self):
        super().__init__('serial_controller')
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_states_callback,
            10)
        self.steering_publisher = self.create_publisher(Float64, '/servo_steering_controller/command', 10)
        self.wheel_publisher = self.create_publisher(Float64, '/rear_wheel_controller/command', 10)
        self.encoder_publisher = self.create_publisher(Float64, '/encoder_vals', 10)  # Publisher for encoder values

        self.wheelbase_length = 0.26
        self.wheelbase_width = 0.213

        self.ben_publisher = self.create_publisher(Bool, '/lift', 10)
        
        self.joy_subscription = self.create_subscription(
            Joy,
            '/joy',
            self.joy_callback,
            10)

        self.declare_parameter('serial_port', value="/dev/ttyAMA10")
        self.serial_port_name = self.get_parameter('serial_port').value

        self.declare_parameter('baud_rate', value=460800)
        self.baud_rate = self.get_parameter('baud_rate').value

        self.declare_parameter('serial_debug', value=False)
        self.debug_serial_cmds = self.get_parameter('serial_debug').value
        if self.debug_serial_cmds:
            self.get_logger().info("Serial debug enabled")

        self.get_logger().info(f"Connecting to port {self.serial_port_name} at {self.baud_rate}.")
        self.conn = serial.Serial(self.serial_port_name, self.baud_rate, timeout=0.2)
        self.get_logger().info(f"Connected to {self.conn.port}")
        self.mutex = Lock()

        self.last_steering_angle = 0.0
        self.last_wheel_velocity = 0.0
        self.ben_state = False
        self.last_button_state = 0
        self.last_encoder_read_time = time.time()
        self.encoder_data = 0

        self.create_timer(0.2, self.read_encoder_data)  

    def joint_states_callback(self, msg):
        try:
            steering_joint_name = "virtual_front_wheel_joint"
            wheel_joint_name = "virtual_rear_wheel_joint"

            steering_index = msg.name.index(steering_joint_name)
            wheel_index = msg.name.index(wheel_joint_name)

            current_steering_angle = msg.position[steering_index]
            current_wheel_velocity = msg.velocity[wheel_index]

            steering_msg = Float64()
            steering_msg.data = current_steering_angle
            self.steering_publisher.publish(steering_msg)

            wheel_msg = Float64()
            wheel_msg.data = current_wheel_velocity
            self.wheel_publisher.publish(wheel_msg)

            if current_steering_angle != self.last_steering_angle or current_wheel_velocity != self.last_wheel_velocity:
                self.send_serial_data(current_steering_angle, current_wheel_velocity)
                self.last_steering_angle = current_steering_angle
                self.last_wheel_velocity = current_wheel_velocity

        except ValueError as e:
            self.get_logger().error(f"Joint name not found: {e}")
    def joy_callback(self, msg):
        current_button_state = msg.buttons[4]
        if current_button_state == 1 and self.last_button_state == 0:  
            self.ben_state = not self.ben_state  
            self.send_command(f"{'2' if self.ben_state else '3'}\n")  
            self.get_logger().info(f"Truck lift: {'On' if self.ben_state else 'Off'}")  
            self.ben_publisher.publish(Bool(data=self.ben_state))  

        self.last_button_state = current_button_state  
    def send_command(self, cmd_string):
        self.mutex.acquire()
        try:
            cmd_string += "\r"
            self.conn.write(cmd_string.encode("utf-8"))
        
        finally:
            self.mutex.release()
    def close_conn(self):
        self.conn.close()

    def send_serial_data(self, steering_angle, wheel_velocity):
        data = f"{steering_angle:.3f};{wheel_velocity:.3f}\n"
        self.send_command(data)
        self.get_logger().info(f"Sent data to serial: {data.strip()}")
    def read_encoder_data(self):
        self.mutex.acquire()
        try:
            # Read encoder data from the serial port
            if self.conn.in_waiting > 0:
                encoder_data = self.conn.readline().decode('utf-8').strip()
                if encoder_data:
                    try:
                        encoder_value = float(encoder_data)
                        encoder_msg = Float64()
                        encoder_msg.data = encoder_value
                        self.encoder_publisher.publish(encoder_msg)
                        self.get_logger().info(f"Encoder data: {encoder_value}")
                    except ValueError:
                        self.get_logger().error(f"Failed to parse encoder data: {encoder_data}")
        finally:
            self.mutex.release()
def main(args=None):
    rclpy.init(args=args)
    serial_controller = SerialController()
    rclpy.spin(serial_controller)
    serial_controller.destroy_node()
    serial_controller.close_conn()
    rclpy.shutdown()

if _name_ == '_main_':
    main()
