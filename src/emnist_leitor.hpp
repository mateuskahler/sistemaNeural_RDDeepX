/*
 * Este arquivo contém uma estrutura para guardar os objetos dos bancos do EMNIST e
 * função pra lê-los de arquivo
 * 
 * Mateus Kahler, Setembro/2018
 * */
  
#ifndef EMNIST_LEITOR_INCLUSO
#define EMNIST_LEITOR_INCLUSO sim

#include <array>
#include <vector>
#include <cstdint>

using std::int8_t;  //inteiro de 8 bits com sinal
using std::uint8_t; //inteiro de 8 bits sem sinal (aka byte)

//define tipo para bitmap de 28 x 28 bytes
	typedef std::array<std::array<std::uint8_t, 28>, 28> bm_ref_t;

//dígito de referência: bitmap associado a etiqueta
	struct dig_ref_t
	{
		bm_ref_t img;      //bitmap (28x28, 8 bits por pixel)
		int8_t   etiqueta; //etiqueta correta
	};

//define tipo vetor de referências
	typedef std::vector<dig_ref_t> dig_refs_t; 

//carrega e retorna referências de arquivo, junto com string de status
std::pair<dig_refs_t, std::string> EMNIST_carrega_referencias(
	std::string arq_imgs, //caminho para arquivo de imagens
	std::string arq_etqs,  //caminho para arquivo de etiquetas/labels
	bool transpoe_img = true //troca linhas/coluna das imagens
	);
	
#endif // fim de header guard MNIST_LEITOR_INCLUSO