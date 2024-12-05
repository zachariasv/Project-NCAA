import subprocess
import os
import logging
import platform

def main():
    # Clear the terminal screen
    os.system("cls" if platform.system() == "Windows" else "clear")
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Get script directory and move up one level to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    scripts = [
        "Match Results/fetch_match_results.py",
        "Match Results/update_team_scores.py",
        "Prediction/feature_engineering.py"
    ]
    
    for script in scripts:
        logging.info(f"Running {script.split('/')[-1]}...")
        result = subprocess.run(["python", os.path.join(project_dir, script)], check=True)
        if result.returncode != 0:
            logging.info(f"Error running {script}")
            return
    
    logging.info("All datasets updated successfully!")

if __name__ == "__main__":
    main()