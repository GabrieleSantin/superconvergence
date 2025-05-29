set -e
export BASEDIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ;  pwd -P)"
cd "${BASEDIR}"

# Create and source virtualenv
if [ -e "${BASEDIR}/venv/bin/activate" ]; then
	echo "using existing virtualenv"
else	
	echo "creating virtualenv ..."
	virtualenv --python=python3 venv
fi

source venv/bin/activate

# Upgrade pip and install libraries (dependencies are also installed)
pip install --upgrade pip
pip install git+https://github.com/GabrieleSantin/VKOGA.git
pip install matplotlib
