import subprocess
import pkg_resources
import sys

def check_and_install_package():
    try:
        # Try to get the installed version
        pkg_resources.get_distribution('RUSH')
        
        # Check if it's installed in editable mode by checking the .egg-link file
        RUSH_egg_link = None
        for path in pkg_resources.working_set.entries:
            if 'RUSH.egg-link' in str(path):
                RUSH_egg_link = path
                break
        
        if not RUSH_egg_link:
            print("RUSH is installed but not in editable mode. Reinstalling...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', 'RUSH', '-y'])
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e', '.'])
            print("Reinstalled RUSH in editable mode.")
        else:
            print("RUSH is already installed in editable mode.")
            
    except pkg_resources.DistributionNotFound:
        print("RUSH not found. Installing...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e', '.'])
        print("Installed RUSH in editable mode.")

if __name__ == "__main__":
    check_and_install_package()