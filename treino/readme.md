
### Demonstração de treino e teste de rede neural

Esta pasta possui quatro programas, para testar e treinar reconhecimento de letras e números.
A execução dos programas requer que o arquivo _emnist-balanced.7z_ seja descompactado (o GitHub não aceitou os arquivos descompactados, muito grandes).

Para compilar o programa de treinamento do reconhecimento de números:

 >  `g++ -o treinoN treino_numeros.cpp ../src/emnist_leitor.cpp ../src/sisneu.cpp -I../src -O2`

_(É necessário fornecer caminho para incluir "sisneu.hpp" e linkar com "sisneu.cpp". A linha acima mostra exemplo disto no GCC, com a estrutura de pastas fornecidas no repositório.)_

Ao ser executado, deve imprimir no console:
  >  `112800 imagens carregadas`\
  >  `iniciando treino...`\
  >  `fim de epoca 1 de 3`\
  >  `fim de epoca 2 de 3`\
  >  `fim de epoca 3 de 3`\
  >   `resultado salvo em 'sistema_treinado_numeros'`
 
Depois de executado, curioso do desempenho da rede, deve-se compilar o programa de teste: 

 >  ` g++ -o testeN teste_numeros.cpp ../src/emnist_leitor.cpp ../src/sisneu.cpp -I../src -O2`

O programa de teste deve imprimir:
  > ` 18800 imagens carregadas`\
  > ` Amostras usadas em teste  : 4000`\
  > ` Acertos: 92.4%`\
  > ` Classe  0 : 95.0% de 400`\
  > ` Classe  1 : 96.8% de 400`\
  > ` Classe  2 : 90.3% de 400`\
  > ` Classe  3 : 90.8% de 400`\
  > ` Classe  4 : 92.5% de 400`\
  > ` Classe  5 : 86.0% de 400`\
  > ` Classe  6 : 94.8% de 400`\
  > ` Classe  7 : 93.5% de 400`\
  > ` Classe  8 : 92.5% de 400`\
  > ` Classe  9 : 91.8% de 400`

O programa de treino para identificação de letras pode ser compilado com

 >  `g++ -o treinoL treino_letras.cpp ../src/emnist_leitor.cpp ../src/sisneu.cpp -I../src -O2`
 
 E o respectivo teste:
  
 >  `g++ -o testeL teste_letras.cpp ../src/emnist_leitor.cpp ../src/sisneu.cpp -I../src -O2`
 
 Para uma descrição algorítmos e referências consultadas, é fornecido [relatório técnico](../relatório_técnico.pdf).
 
