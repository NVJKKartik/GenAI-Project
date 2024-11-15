#!/usr/bin/env bash
cd "$(dirname "${BASH_SOURCE[0]}")"

USE_VENV=1
delimiter="################################################################"
VENV_DIR="./venv"

if [[ "$1" == 'T' ]]
then
    printf "Cleaning virtual env folder."    
    rm -rf venv >stdout.txt 2>stderr.txt
fi

if [[ "$2" == 'F' ]]
then
    USE_VENV=0
    printf "You chose not to use venv..." 
fi

if [[ -z "${python_cmd}" ]]
then
    python_cmd="python3"
fi

# Verify python installation
if [[ "$(${python_cmd} -V)" =~ "Python 3" ]] &> /dev/null
then
    printf "Python 3 is installed" 
    printf "\n%s\n" "${delimiter}"
else
    printf "Python 3 is not installed. Please install Python 3.10.6 before continue.."     
    printf "\n%s\n" "${delimiter}"
    exit 1	
fi

if [[ $USE_VENV -eq 1 ]] && ! "${python_cmd}" -c "import venv" &>/dev/null
then
    printf "\n%s\n" "${delimiter}"
    printf "\e[1m\e[31mERROR: python3-venv is not installed, aborting...\e[0m"
    printf "\n%s\n" "${delimiter}"
    exit 1
fi

if [[ $USE_VENV -eq 1 ]] && [[ -z "${VIRTUAL_ENV}" ]];
then    
    if [[ ! -d "${VENV_DIR}" ]]
    then
        printf "\n%s\n" "${delimiter}"
        printf "Creating python venv"
        printf "\n%s\n" "${delimiter}"
        "${python_cmd}" -m venv "${VENV_DIR}"        
    fi
    
    if [[ -f "${VENV_DIR}"/bin/activate ]]
    then        
        printf "Activating python venv"        
        source venv/bin/activate        
    else
        printf "\n%s\n" "${delimiter}"
        printf "\e[1m\e[31mERROR: Cannot activate python venv, aborting...\e[0m"
        printf "\n%s\n" "${delimiter}"
        exit 1
    fi
else
    printf "\n%s\n" "${delimiter}"
    printf "python venv already activate or run without venv. ${VIRTUAL_ENV}"
    printf "\n%s\n" "${delimiter}"
fi

printf "\n%s\n" "${delimiter}"
printf "Installing requirements. This could take a few minutes..."
pip install -r requirements.txt
printf "\n%s\n" "${delimiter}"
printf "All done! Launch 'start.sh'"
printf "\n%s\n" "${delimiter}"
