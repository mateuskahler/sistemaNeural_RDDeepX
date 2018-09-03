### Demonstração do uso da rede neural treinada para identificar caracteres escritos a mão.

Para compilar:

`g++ -o exemplo exemplo.cpp ../src/sisneu.cpp -I../src`

_(É necessário fornecer caminho para incluir "sisneu.hpp" e linkar com "sisneu.cpp". A linha acima mostra exemplo disto no GCC, com a estrutura de pastas fornecidas no repositório.)_

Ao ser executado, o programa carrega as imagens fornecidas e interpreta cada caractere, imprimindo-os no console.
