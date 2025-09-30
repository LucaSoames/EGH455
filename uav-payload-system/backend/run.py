#!/usr/bin/env python3
"""
UAV Payload Tracking and Acquisition System (TAQ-25)
Main application entry point
"""

import os
from dotenv import load_dotenv
from app import create_app, socketio, db
from app.models import User, UAV, TelemetryData, SystemLog
from werkzeug.security import generate_password_hash
from datetime import datetime

# Load environment variables
load_dotenv()

# Create Flask application
app = create_app()

@app.cli.command()
def init_db():
    """Initialize the database with sample data"""
    print("Initializing database...")
    
    # Create all tables
    db.create_all()
    
    # Create admin user if not exists
    admin_user = User.query.filter_by(username='admin').first()
    if not admin_user:
        admin_user = User(
            username='admin',
            email='admin@uavsystem.com',
            password_hash=generate_password_hash('admin123'),
            role='admin',
            is_active=True
        )
        db.session.add(admin_user)
        print("Created admin user (username: admin, password: admin123)")
    
    # Create sample operator user
    operator_user = User.query.filter_by(username='operator').first()
    if not operator_user:
        operator_user = User(
            username='operator',
            email='operator@uavsystem.com',
            password_hash=generate_password_hash('operator123'),
            role='operator',
            is_active=True
        )
        db.session.add(operator_user)
        print("Created operator user (username: operator, password: operator123)")
    
    # Create sample UAVs if not exist
    if UAV.query.count() == 0:
        sample_uavs = [
            UAV(
                serial_number='UAV-001',
                model='DJI Matrice 300 RTK',
                max_payload_weight=2.7,
                max_altitude=7000,
                max_speed=23.0,
                battery_capacity=5935,
                communication_range=15000,
                status='active'
            ),
            UAV(
                serial_number='UAV-002',
                model='DJI Phantom 4 RTK',
                max_payload_weight=1.0,
                max_altitude=6000,
                max_speed=20.0,
                battery_capacity=5870,
                communication_range=7000,
                status='active'
            ),
            UAV(
                serial_number='UAV-003',
                model='Autel EVO Max 4T',
                max_payload_weight=1.2,
                max_altitude=5000,
                max_speed=15.0,
                battery_capacity=7100,
                communication_range=12000,
                status='maintenance'
            )
        ]
        
        for uav in sample_uavs:
            db.session.add(uav)
        print(f"Created {len(sample_uavs)} sample UAVs")
    
    # Sample payloads removed for minimal system
    
    # Commit all changes
    db.session.commit()
    print("Database initialization complete!")

@app.cli.command()
def create_admin():
    """Create an admin user"""
    username = input("Enter admin username: ")
    email = input("Enter admin email: ")
    password = input("Enter admin password: ")
    
    # Check if user already exists
    existing_user = User.query.filter(
        (User.username == username) | (User.email == email)
    ).first()
    
    if existing_user:
        print("User with this username or email already exists!")
        return
    
    admin_user = User(
        username=username,
        email=email,
        password_hash=generate_password_hash(password),
        role='admin',
        is_active=True
    )
    
    db.session.add(admin_user)
    db.session.commit()
    
    print(f"Admin user '{username}' created successfully!")

@app.shell_context_processor
def make_shell_context():
    """Make database models available in Flask shell"""
    return {
        'db': db,
        'User': User,
        'UAV': UAV,
        'TelemetryData': TelemetryData,
        'SystemLog': SystemLog
    }

if __name__ == '__main__':
    # Development server
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    socketio.run(
        app,
        host='0.0.0.0',
        port=5000,
        debug=debug_mode,
        allow_unsafe_werkzeug=True  # Always allow for development
    )
