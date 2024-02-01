######################################################################
# Customization section
######################################################################

poetry:
	python3 -m pip install pipx
	pipx install poetry
	poetry env use python3
	poetry install

pip:
	python3 -m pip install pipreqs
	pipreqs --force --savepath requirements.txt
	python3 -m pip install -r requirements.txt

pip_env:
	python3 -m venv env
	. env/bin/activate
	python3 -m pip install pipreqs
	pipreqs --force --savepath requirements.txt
	python3 -m pip install -r requirements.txt
	