import argparse
import sys

from streamlit.web import cli as stcli

# TODO: use theses flags to run the app if there will be problems with the streamlit cli
# "--browser.serverAddress", "localhost",
# "--server.enableWebsocketCompression", "false",
# "--server.enableCORS", "false",
# "--server.enableXsrfProtection", "false",
# "global.logLevel", "error",
# "global.suppressWarning", "true",
# "global.sharingMode", "off"]
# add python path to the sys.path
# file_path = Path(__file__).parents[2]
# sys.path.append(str(file_path))
# add python path to the sys.path

if __name__ == '__main__':
    # use argparse to get the flags
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_port', default=None, type=str,
                        help='add port to run the app. for example: --server_port 8501')
    args = parser.parse_args()

    sys.argv = ["streamlit", "run", "VisualizeResults.py",
                # "--server.headless", "true",
                # "global.developmentMode", "false",
                ]
    if args.server_port:
        sys.argv.append("--server.port")
        sys.argv.append(args.server_port)

    sys.exit(stcli.main())
