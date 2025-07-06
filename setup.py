#!/usr/bin/env python3
"""
3db Unified Database Ecosystem - Setup Script

This script helps users quickly set up and configure the 3db system
with automatic dependency checking, database initialization, and validation.
"""

import os
import sys
import subprocess
import shutil
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Setup3db:
    """3db Setup and Configuration Manager."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.requirements_checked = False
        self.docker_available = False
        self.python_version_ok = False
        
    def print_header(self):
        """Print the 3db setup header."""
        print(f"{Colors.HEADER}{Colors.BOLD}")
        print("╔" + "═" * 60 + "╗")
        print("║" + " " * 15 + "🧠 3db Unified Database Ecosystem" + " " * 12 + "║")
        print("║" + " " * 20 + "Setup & Configuration Tool" + " " * 17 + "║")
        print("╚" + "═" * 60 + "╝")
        print(f"{Colors.ENDC}")
        print()
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        print(f"{Colors.OKBLUE}🐍 Checking Python version...{Colors.ENDC}")
        
        version = sys.version_info
        if version.major == 3 and version.minor >= 11:
            print(f"{Colors.OKGREEN}✅ Python {version.major}.{version.minor}.{version.micro} is compatible{Colors.ENDC}")
            self.python_version_ok = True
            return True
        else:
            print(f"{Colors.FAIL}❌ Python {version.major}.{version.minor}.{version.micro} is not compatible{Colors.ENDC}")
            print(f"{Colors.WARNING}   Required: Python 3.11 or higher{Colors.ENDC}")
            return False
    
    def check_docker(self) -> bool:
        """Check if Docker and Docker Compose are available."""
        print(f"{Colors.OKBLUE}🐳 Checking Docker installation...{Colors.ENDC}")
        
        try:
            # Check Docker
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, check=True)
            docker_version = result.stdout.strip()
            
            # Check Docker Compose
            result = subprocess.run(['docker-compose', '--version'], 
                                  capture_output=True, text=True, check=True)
            compose_version = result.stdout.strip()
            
            print(f"{Colors.OKGREEN}✅ {docker_version}{Colors.ENDC}")
            print(f"{Colors.OKGREEN}✅ {compose_version}{Colors.ENDC}")
            self.docker_available = True
            return True
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"{Colors.FAIL}❌ Docker or Docker Compose not found{Colors.ENDC}")
            print(f"{Colors.WARNING}   Please install Docker Desktop or Docker Engine{Colors.ENDC}")
            print(f"{Colors.WARNING}   Visit: https://docs.docker.com/get-docker/{Colors.ENDC}")
            return False
    
    def check_git(self) -> bool:
        """Check if Git is available."""
        print(f"{Colors.OKBLUE}📦 Checking Git installation...{Colors.ENDC}")
        
        try:
            result = subprocess.run(['git', '--version'], 
                                  capture_output=True, text=True, check=True)
            git_version = result.stdout.strip()
            print(f"{Colors.OKGREEN}✅ {git_version}{Colors.ENDC}")
            return True
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"{Colors.WARNING}⚠️ Git not found (optional for local setup){Colors.ENDC}")
            return False
    
    def create_env_file(self) -> bool:
        """Create .env file from template if it doesn't exist."""
        print(f"{Colors.OKBLUE}⚙️ Setting up environment configuration...{Colors.ENDC}")
        
        env_file = self.project_root / '.env'
        env_example = self.project_root / 'config' / '.env.example'
        
        if env_file.exists():
            print(f"{Colors.WARNING}⚠️ .env file already exists, skipping creation{Colors.ENDC}")
            return True
        
        if not env_example.exists():
            print(f"{Colors.FAIL}❌ .env.example template not found{Colors.ENDC}")
            return False
        
        try:
            shutil.copy2(env_example, env_file)
            print(f"{Colors.OKGREEN}✅ Created .env file from template{Colors.ENDC}")
            print(f"{Colors.WARNING}⚠️ Please edit .env file with your database passwords{Colors.ENDC}")
            return True
            
        except Exception as e:
            print(f"{Colors.FAIL}❌ Failed to create .env file: {e}{Colors.ENDC}")
            return False
    
    def install_python_dependencies(self, dev: bool = False) -> bool:
        """Install Python dependencies."""
        print(f"{Colors.OKBLUE}📚 Installing Python dependencies...{Colors.ENDC}")
        
        try:
            # Install main dependencies
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                          check=True, cwd=self.project_root)
            print(f"{Colors.OKGREEN}✅ Main dependencies installed{Colors.ENDC}")
            
            # Install development dependencies if requested
            if dev:
                subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-dev.txt'], 
                              check=True, cwd=self.project_root)
                print(f"{Colors.OKGREEN}✅ Development dependencies installed{Colors.ENDC}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"{Colors.FAIL}❌ Failed to install dependencies: {e}{Colors.ENDC}")
            return False
    
    def setup_database_schema(self) -> bool:
        """Set up database schemas."""
        print(f"{Colors.OKBLUE}🗄️ Setting up database schemas...{Colors.ENDC}")
        
        schema_files = [
            self.project_root / 'schemas' / 'postgresql_schema.sql',
            self.project_root / 'schemas' / 'age_graph_schema.sql'
        ]
        
        for schema_file in schema_files:
            if not schema_file.exists():
                print(f"{Colors.FAIL}❌ Schema file not found: {schema_file}{Colors.ENDC}")
                return False
        
        print(f"{Colors.OKGREEN}✅ Database schema files are ready{Colors.ENDC}")
        print(f"{Colors.WARNING}   Schemas will be applied automatically when Docker containers start{Colors.ENDC}")
        return True
    
    def start_docker_services(self, profile: str = "basic") -> bool:
        """Start Docker services based on profile."""
        print(f"{Colors.OKBLUE}🚀 Starting Docker services ({profile} profile)...{Colors.ENDC}")
        
        profiles = {
            "basic": ["postgresql-main", "redis"],
            "development": ["--profile", "development", "--profile", "admin-tools"],
            "production": ["--profile", "separate-databases", "--profile", "monitoring"],
            "full": ["--profile", "separate-databases", "--profile", "monitoring", "--profile", "admin-tools"]
        }
        
        try:
            if profile == "basic":
                cmd = ['docker-compose', 'up', '-d'] + profiles[profile]
            else:
                cmd = ['docker-compose'] + profiles[profile] + ['up', '-d']
            
            print(f"{Colors.OKCYAN}   Executing: {' '.join(cmd)}{Colors.ENDC}")
            
            result = subprocess.run(cmd, cwd=self.project_root, 
                                  capture_output=True, text=True, check=True)
            
            print(f"{Colors.OKGREEN}✅ Docker services started successfully{Colors.ENDC}")
            
            # Wait for services to be ready
            print(f"{Colors.OKCYAN}   Waiting for services to initialize...{Colors.ENDC}")
            time.sleep(10)
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"{Colors.FAIL}❌ Failed to start Docker services: {e}{Colors.ENDC}")
            if e.stderr:
                print(f"{Colors.FAIL}   Error: {e.stderr}{Colors.ENDC}")
            return False
    
    def verify_services(self) -> bool:
        """Verify that services are running correctly."""
        print(f"{Colors.OKBLUE}🔍 Verifying service health...{Colors.ENDC}")
        
        services_to_check = [
            ("PostgreSQL", "postgresql-main", 5432),
            ("Redis", "redis", 6379)
        ]
        
        all_healthy = True
        
        for service_name, container_name, port in services_to_check:
            try:
                # Check if container is running
                result = subprocess.run(['docker', 'ps', '--filter', f'name={container_name}', '--format', 'table {{.Names}}'], 
                                      capture_output=True, text=True, check=True)
                
                if container_name in result.stdout:
                    print(f"{Colors.OKGREEN}✅ {service_name} container is running{Colors.ENDC}")
                else:
                    print(f"{Colors.FAIL}❌ {service_name} container is not running{Colors.ENDC}")
                    all_healthy = False
                    
            except subprocess.CalledProcessError:
                print(f"{Colors.FAIL}❌ Failed to check {service_name} status{Colors.ENDC}")
                all_healthy = False
        
        return all_healthy
    
    def test_api_connection(self) -> bool:
        """Test connection to the 3db API if it's running."""
        print(f"{Colors.OKBLUE}🌐 Testing API connection...{Colors.ENDC}")
        
        try:
            import requests
            response = requests.get('http://localhost:8000/health', timeout=10)
            
            if response.status_code == 200:
                print(f"{Colors.OKGREEN}✅ API is responding correctly{Colors.ENDC}")
                health_data = response.json()
                if health_data.get('overall_health'):
                    print(f"{Colors.OKGREEN}✅ All components are healthy{Colors.ENDC}")
                else:
                    print(f"{Colors.WARNING}⚠️ Some components may have issues{Colors.ENDC}")
                return True
            else:
                print(f"{Colors.FAIL}❌ API returned status code: {response.status_code}{Colors.ENDC}")
                return False
                
        except ImportError:
            print(f"{Colors.WARNING}⚠️ requests library not available, skipping API test{Colors.ENDC}")
            return True
        except Exception as e:
            print(f"{Colors.WARNING}⚠️ API not available yet (this is normal): {e}{Colors.ENDC}")
            return True
    
    def run_example_demo(self) -> bool:
        """Run the example demo to verify everything works."""
        print(f"{Colors.OKBLUE}🎯 Running example demo...{Colors.ENDC}")
        
        demo_file = self.project_root / 'examples' / 'complete_demo.py'
        
        if not demo_file.exists():
            print(f"{Colors.FAIL}❌ Demo file not found: {demo_file}{Colors.ENDC}")
            return False
        
        try:
            print(f"{Colors.OKCYAN}   This may take a moment...{Colors.ENDC}")
            result = subprocess.run([sys.executable, str(demo_file)], 
                                  cwd=self.project_root, 
                                  capture_output=True, text=True, 
                                  timeout=120)  # 2 minute timeout
            
            if result.returncode == 0:
                print(f"{Colors.OKGREEN}✅ Demo completed successfully!{Colors.ENDC}")
                return True
            else:
                print(f"{Colors.FAIL}❌ Demo failed with return code: {result.returncode}{Colors.ENDC}")
                if result.stderr:
                    print(f"{Colors.FAIL}   Error output: {result.stderr[:500]}...{Colors.ENDC}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"{Colors.WARNING}⚠️ Demo timed out (may still be working){Colors.ENDC}")
            return True
        except Exception as e:
            print(f"{Colors.FAIL}❌ Failed to run demo: {e}{Colors.ENDC}")
            return False
    
    def print_next_steps(self, profile: str):
        """Print next steps and useful information."""
        print(f"\n{Colors.HEADER}{Colors.BOLD}🎉 3db Setup Complete!{Colors.ENDC}\n")
        
        print(f"{Colors.OKGREEN}📋 Available Services:{Colors.ENDC}")
        print(f"   🌐 API Documentation: http://localhost:8000/docs")
        print(f"   🏥 Health Check: http://localhost:8000/health")
        print(f"   📊 Metrics: http://localhost:8000/metrics")
        
        if profile in ["development", "full"]:
            print(f"   🔧 pgAdmin: http://localhost:5050 (admin@3db.local / admin123)")
            print(f"   📊 Redis Commander: http://localhost:8081 (admin / admin123)")
        
        if profile in ["production", "full"]:
            print(f"   📈 Grafana: http://localhost:3000 (admin / admin123)")
            print(f"   🎯 Prometheus: http://localhost:9090")
        
        print(f"\n{Colors.OKBLUE}🚀 Quick Start Commands:{Colors.ENDC}")
        print(f"   # Check system status")
        print(f"   curl http://localhost:8000/health")
        print()
        print(f"   # Create your first entity")
        print(f"   curl -X POST 'http://localhost:8000/entities/' \\")
        print(f"     -H 'Content-Type: application/json' \\")
        print(f"     -d '{{\"entity_type\": \"user\", \"data\": {{\"name\": \"Alice\", \"email\": \"alice@example.com\"}}}}'")
        print()
        print(f"   # Run the complete demo")
        print(f"   python examples/complete_demo.py")
        
        print(f"\n{Colors.WARNING}⚠️ Important Notes:{Colors.ENDC}")
        print(f"   • Edit .env file with your secure passwords")
        print(f"   • Database data is persisted in Docker volumes")
        print(f"   • Use 'docker-compose down -v' to completely reset data")
        print(f"   • Check logs with 'docker-compose logs -f [service-name]'")
        
        print(f"\n{Colors.HEADER}📚 Documentation: README.md{Colors.ENDC}")
        print(f"{Colors.HEADER}🤝 Support: GitHub Issues{Colors.ENDC}")
        print()
    
    def setup(self, profile: str = "basic", dev: bool = False, skip_demo: bool = False) -> bool:
        """Run the complete setup process."""
        self.print_header()
        
        print(f"{Colors.OKBLUE}Starting 3db setup with '{profile}' profile...{Colors.ENDC}\n")
        
        # Check prerequisites
        if not self.check_python_version():
            return False
        
        if not self.check_docker():
            return False
        
        self.check_git()  # Optional
        
        print()
        
        # Setup configuration
        if not self.create_env_file():
            return False
        
        # Install dependencies
        if not self.install_python_dependencies(dev=dev):
            return False
        
        # Setup database schemas
        if not self.setup_database_schema():
            return False
        
        print()
        
        # Start Docker services
        if not self.start_docker_services(profile):
            return False
        
        # Verify services
        if not self.verify_services():
            print(f"{Colors.WARNING}⚠️ Some services may not be ready yet, this is normal{Colors.ENDC}")
        
        # Test API connection (if applicable)
        if profile in ["development", "full"]:
            self.test_api_connection()
        
        # Run demo (if not skipped)
        if not skip_demo and profile != "production":
            print()
            self.run_example_demo()
        
        # Print next steps
        self.print_next_steps(profile)
        
        return True


def main():
    """Main setup function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="3db Unified Database Ecosystem Setup Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup.py                          # Basic setup with PostgreSQL and Redis
  python setup.py --profile development    # Development setup with admin tools
  python setup.py --profile production     # Production setup with monitoring
  python setup.py --profile full           # Complete setup with all features
  python setup.py --dev                    # Install development dependencies
  python setup.py --skip-demo              # Skip running the demo
        """
    )
    
    parser.add_argument(
        '--profile', 
        choices=['basic', 'development', 'production', 'full'],
        default='basic',
        help='Setup profile to use (default: basic)'
    )
    
    parser.add_argument(
        '--dev',
        action='store_true',
        help='Install development dependencies'
    )
    
    parser.add_argument(
        '--skip-demo',
        action='store_true',
        help='Skip running the example demo'
    )
    
    args = parser.parse_args()
    
    # Create setup instance and run
    setup = Setup3db()
    success = setup.setup(
        profile=args.profile,
        dev=args.dev,
        skip_demo=args.skip_demo
    )
    
    exit_code = 0 if success else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
