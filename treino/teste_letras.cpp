/**
 * Este programa testa uma rede treinada para identificar letras
 * 
 * para compilar:
 * 
 *   g++ -o testeL teste_letras.cpp ../src/emnist_leitor.cpp ../src/sisneu.cpp -I../src -O2
 *
 *   É necessário fornecer caminho para incluir "sisneu.hpp" e linkar com "sisneu.cpp".
 * A linha acima mostra exemplo disto no GCC, com a estrutura de pastas fornecidas no repositório.
 * O programa requer o banco de teste, que é fornecido no repositório e deve ser descompactado na pasta do executável.
 *
 *  O programa procura o sistema neural em um arquivo chamado "sistema_treinado_letras",
 * como o gerado pelo programa disponível em "treino_letras.cpp"
 */
 
// inclui classes desenvolvidas para redes neurais
	#include "sisneu.hpp"

//bibliotecas padrão requeridas
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>

//nome dos arquivos do banco de teste
	std::string const arq_img("emnist-balanced-test-images-idx3-ubyte");
	std::string const arq_etq("emnist-balanced-test-labels-idx1-ubyte");
	
//nome do arquivo com o sistema neural treinado
	std::string const arq_sisL("sistema_treinado_letras");

//função utilitária para imprimir resultado de teste no console
	std::ostream& operator<< (std::ostream& os, const teste_resultado_t & T);

int main(int, char**)
{
	//tenta carregar amostras de banco de teste
		classes_separadas_t amostras_teste;
		{
			auto carrega = EMNIST_carrega_referencias(arq_img, arq_etq);
			std::cout << carrega.second << std::endl; // imprime mensagem retornada pelo leitor do arquivo (ex.: "60000 amostras carregadas")
			if(carrega.first.empty()) // se vetor vazio: falha de leitura (ex.: arquivo não encontrado)
			{
				std::cout << "Ocorreu um erro ao carregar banco. Abortando."<< std::endl;
				return 1;
			}
			amostras_teste = classes_separadas_t(std::move(carrega.first)); //usa vetor lido de arquivo para inicializar gerar classes de teste
		}
		
	//exclui caracteres numéricos (10 primeiras classes)
		for(int k=0; k<10; k++)
			{ amostras_teste.exclui_classe(0);	}
			
	//carrega sistema neural de arquivo
		sistema_neural_t S;
		if(!S.carrega_de_arquivo(arq_sisL))
		{
			std::cout << "Ocorreu um erro ao carregar sistema treinado. Abortando."<< std::endl;
			std::cout << "existe um arquivo de sistema treinado nesta pasta?"<< std::endl;
			return 1;
		}
	
	//testa sistema contra banco
		auto T = testa_sistema(S, amostras_teste);
		std::cout << T << std::endl;
	return 0;
}

//implementação de função utilitária para imprimir resultado de teste no console
std::ostream& operator<< (std::ostream& os, const teste_resultado_t & T) 
{
	//lambda para formatar porcentagem (são mais confortáveis de se ler)
		auto formata_por_cento = [](float p)->std::string
			{
				int im = ((p*1000.f)+0.5f);
				int ic = im%10;
				im /= 10;
				std::stringstream ss;
				ss << im << '.' << ic << '%';
				return ss.str();
			};
	//resultado geral
		os << "Amostras usadas em teste  : " << T.N_tot << std::endl;
		float accT = T.N_acertos_tot;
		accT /= T.N_tot;
		os << "Acertos: " << formata_por_cento(accT) << std::endl;
	//resultado por classe
		vec_f accC(T.N_classes, 0.f);
		for(int k=0; k<T.N_classes; k++)
		{
			float accC = T.N_acertos_classe[k];
			accC /= T.N_amostras_classe[k];
			os << "Classe " << std::setw(2) << k << " : " << std::setw(3) << formata_por_cento(accC) << " de " <<  T.N_amostras_classe[k] << std::endl;
		}
	return os;
}