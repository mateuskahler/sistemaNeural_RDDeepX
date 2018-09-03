/**
 * Este programa demonstra treino da rede para reconhecer números
 * 
 * para compilar:
 *
 *   g++ -o treinoN treino_numeros.cpp ../src/emnist_leitor.cpp ../src/sisneu.cpp -I../src -O2
 *
 *   É necessário fornecer caminho para incluir "sisneu.hpp" e linkar com "sisneu.cpp".
 * A linha acima mostra exemplo disto no GCC, com a estrutura de pastas fornecidas no repositório.
 * O programa requer o banco de treino, que é fornecido no repositório e deve ser descompactado na pasta do executável.
 * 
 * Habilitar algum nível de otimização é uma boa idéia, já que a tarefa é computacionalmente intensa.
 * A rede treinada será salva no arquivo "sistema_treinado_numeros", que pode ser testado com o programa
 * fornecido em "teste_numeros.cpp"
 */
 
// inclui classes desenvolvidas para redes neurais
	#include "sisneu.hpp"

//bibliotecas padrão requeridas
#include <string>
#include <iostream>

//caminho dos arquivos do banco de dados
	std::string const arq_img("emnist-balanced-train-images-idx3-ubyte");
	std::string const arq_etq("emnist-balanced-train-labels-idx1-ubyte");
	
//nome destino para o sistema neural treinado
	std::string const arq_sisN("sistema_treinado_numeros");

int main(int, char**)
{
	//tenta carregar amostras de banco de treino
		classes_separadas_t amostras_treino;
		{
			auto carrega = EMNIST_carrega_referencias(arq_img, arq_etq);
			std::cout << carrega.second << std::endl; // imprime mensagem retornada pelo leitor do arquivo (ex.: "60000 amostras carregadas")
			if(carrega.first.empty()) // se vetor vazio: falha de leitura (ex.: arquivo não encontrado)
			{
				std::cout << "Ocorreu um erro ao carregar banco. Abortando."<< std::endl;
				return 1;
			}
			amostras_treino = classes_separadas_t(std::move(carrega.first)); //usa vetor lido de arquivo para inicializar gerar classes de treino
		}
		
	//mantém apenas classes numéricas (10 primeiras)
		while(amostras_treino.quantidade_classes()>10)
			{ amostras_treino.exclui_classe(10);	}
			
	//cria sistema neural
		sistema_neural_t S;
		int const N_Camadas = 2;//número de camadas
		int const N_Entradas = 28*28; // N entradas = 28*28 pixels
		int const N_Classes = amostras_treino.quantidade_classes(); //N saídas = número de classes
		//neste exemplo, será criado um sistema de 2 camadas, com 30 e 10 neurons
			S.dimensiona(N_Entradas, N_Camadas, vec_i{N_Classes*3, N_Classes});
			S.sorteia_coefs(); //inicializa coeficientes
		//cada época passa pelo banco inteiro de treino
			int const N_Epocas = 3; //3 épocas, para demonstrar
			int const N_por_pacote = N_Classes; //usa o número de classes como tamanho dos mini batches
			float eta    =  3.f; //taxa de aprendizado
			float lambda = 1e-4; //taxa de normalização L2
			std::cout << "iniciando treino..." << std::endl;
			for(int e=0; e<N_Epocas; e++)
			{
				treina_sistema_epoca(S, amostras_treino, N_por_pacote, eta, lambda);
				std::cout << "fim de epoca " << (e+1) << " de " << N_Epocas << std::endl;
			}
		
		//salva resultado para arquivo
			S.salva_para_arquivo(arq_sisN);
			std::cout << " resultado salvo em \'" << arq_sisN << "\' " << std::endl;
	return 0;
}