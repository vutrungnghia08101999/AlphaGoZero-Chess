# Compress Python project to application and connect to Arena GUI

Download models at: https://drive.google.com/drive/folders/1BlMer970u_jLmiJB5wG9l2satDvn6OS9?usp=sharing

Place models in folder models (whatever u want, because we use absolute path)

Compress Python project
- Choose model name in uci.py (use absolute path)
- Choose name of output depend on model name in uci.spec

Create environment
```
conda env create -f env.yml
```

Build command
```
pyinstaller --onefile uci.spec
```

Open Arena and choose intall new engine
