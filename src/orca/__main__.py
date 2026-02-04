import sys
from orca.gui.app import run_gui

def main():
    """
    Main entry point for ORCA.
    If no arguments are provided, launches the GUI.
    """
    # Future CLI argument parsing can go here
    if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] == "--gui"):
        run_gui()
    else:
        print("CLI usage not yet implemented. Use --gui to launch the graphical interface.")

if __name__ == "__main__":
    main()
