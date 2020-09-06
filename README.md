# Detecção de sinais de libras usando YOLO e CNN 

Aplicação para detecção automática de sinais de libras, através da webcan. 
Para detecção são realizadas duas etapas:
1. Utilizaçãod de uma yolo para detecção das mãos 
2. Envio da área delimitada pela YOLO para uma CNN, treinada com um dataset de libras, que faz a previso de qual o sinal mostrado. 

Referências: 
YOLO: https://github.com/cansik/yolo-hand-detection
DATASET libras: http://sites.ecomp.uefs.br/lasic/projetos/libras-dataset
