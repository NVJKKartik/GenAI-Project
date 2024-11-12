export PYTHONWARNINGS=ignore
export CUDA_VISIBLE_DEVICES=0

USE_VENV=1
delimiter="################################################################"
VENV_DIR="./venv"
LANGUAGE="en-US"

if [[ "$1" == "pt-BR" ]]
then
	LANGUAGE="pt-BR"
	printf $LANGUAGE
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
        printf "\n%s\n" "${delimiter}"
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

if [[ $LANGUAGE == 'pt-BR' ]]
then
	python app_pt-br.py
else
	python app.py
fi
