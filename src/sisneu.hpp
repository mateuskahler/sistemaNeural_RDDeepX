/*
 * Este arquivo declara as classes necessárias para o treinamento e teste de uma rede neural
 * 
 * Matrizes, camadas, sistema neural e auxiliares para cálculo de 'back propagation'
 * 
 * Procurou-se nomear as funções explicitamente e comentar passos da implementação,
 * mas o relatório anexo é referência mais formal quanto aos algorítmos
 * 
 * Mateus Kahler, Setembro/2018
 *
 * */
 
#ifndef SISNEU_INCLUSO
#define SISNEU_INCLUSO sim

#include "emnist_leitor.hpp" //leitor desenvolvido para os arquivos do EMNIST

#include <vector>
#include <stdexcept>
#include <functional>
#include <algorithm>

//define o tipo numérico de ponto flutuante utilizado
	typedef float NUM_T;

//define vetores utilitário
	typedef std::vector<NUM_T> vec_f; //vetor de ponto flutuante
	typedef std::vector<int>   vec_i; //vetor de inteiros

// Matriz bidimensional, representa uma matriz através de um vetor de vetores. Cada vetor é uma linha, e cada elemento deste uma coluna
struct Matriz_t;
// Camada Neural, contém uma matriz representando os pesos (w), e um vetor representado o bias (b)
struct camada_neural_t;
// Gradiente de uma camada, contém matriz para dC/dw e vetor para dC/db
struct gradiente_camada_neural_t;
// Gradientess, o gradiente de várias camadas, usado para representar resultado completo de backpropagation
struct gradientes_t;
// Sistema Neural, agrega camadas neurais e fornece métodos para backpropagation e classificação de entrada. Pode ser salvo/carregada de arquivo
struct sistema_neural_t;
// Classes Separadas, organiza amostras em classes e gera y referência para treinamento/teste
struct classes_separadas_t;

// matriz bidimensional
	struct Matriz_t
	{
		Matriz_t()=default;
		int Largura()const;
		int  Altura()const;
		void redimensiona(int largura, int altura); 
		void zera(); //zera todos os elementos
		std::vector<vec_f> mem; //memória
	};

// camada neural, com bias e peso relativo a cada entrada para cada neurônio 
	struct camada_neural_t
	{
		Matriz_t w; //pesos
		vec_f    b; //bias
		void constroi(int N_saidas, int N_entradas); //reserva memória
		vec_f resolve_z(vec_f const & x)const; //calcula e retorna w*x + b
		int n_entradas()const; //retorna número de entradas
		int n_saidas()const; //retorna número de saídas
	};
	
// gradiente de camada neural
	struct gradiente_camada_neural_t
	{
		Matriz_t dCdw; //derivada do custo em relação a w 
		vec_f    dCdz; //derivada do custo em relação a b
		gradiente_camada_neural_t & operator += (gradiente_camada_neural_t const & dG); //soma termo a termo
		gradiente_camada_neural_t & operator *= (NUM_T const k); //multiplicação por escalar
	};
	
// agregado de gradientes de camadas, representando gradiente de sistema
	struct gradientes_t
	{
		std::vector<gradiente_camada_neural_t> gradientes;
		gradientes_t & operator += (gradientes_t const & dG); //soma termo a termo
		gradientes_t & operator *= (NUM_T const k); //multiplicação por escalar
		void zera();
	};

// sistema neural: uma coleção de camadas e alguns ajustes relativos ao número de entradas
	struct sistema_neural_t
	{
		//dimensiona sistema. args: número de entradas, de camadas, e de neurons por camada
			void  dimensiona(int const eN, int const cN, vec_i const cnN);
		//sorteia coeficientes
			void sorteia_coefs();
		//avaliação e backpropagation
			vec_f resolve(vec_f const & x)const;
			void back_prop_inc(vec_f const & x, vec_f const & y, gradientes_t & dest)const;
		//deriva coeficientes (soma elemento-por-elemento e escala)
			sistema_neural_t & deriva_por(gradientes_t const & dS);
		//gera estrutura de gradientes de mesma dimensão que o sistema, com todos valores nulos
			gradientes_t gera_gradiente_nulo()const;
		//funções para salvar e carregar rede em arquivo
			bool salva_para_arquivo(std::string const arq_caminho)const;
			bool carrega_de_arquivo(std::string const arq_caminho);
		//memória
			std::vector<camada_neural_t> camadas;
	};

//estrutura que guarda um par de entrada e saida de sistema
	struct par_x_y_t
	{
		vec_f x;
		vec_f y;
	};
	typedef std::vector<par_x_y_t> pares_x_y;

//estrutura com amostras separadas por classe (para treino balanceado)
	struct classes_separadas_t
	{
		//classe: etiqueta e vetor de amostras
			struct classe_t
			{
				int8_t etiqueta_original;
				vec_f  y_ref;
				vec_i amostras_i;
			};
		//contrói baseado em vetor de referências
			explicit classes_separadas_t(dig_refs_t refs);
			classes_separadas_t() = default;
		int quantidade_classes()const;
		int quantidade_entradas()const;
		int amostras_total()const; //quantidade total de amostras (soma das amostras de todas as classes)
		int amostras_classe(int i)const; //quantidade de amostras em dada classe
		int gera_indice(int classe, int amostra)const;
		vec_f gera_x(int indice)const;
		vec_f gera_y(int indice)const;
		par_x_y_t gera_xy(int indice)const;
		void exclui_classe(int classe_id);
		//memória
		dig_refs_t mem;
		std::vector<classe_t> inds_classificados;
	};

//funções de treino
/**
 * treina_sistema_pacote treina sistema fornecido usando pacote
 * treina_sistema_epoca treina sistema gerando pacotes até exaurir banco fornecido
 * eta e lambda são os 'hiperparâmetros', taxa de aprendizado e de normalização, respectivamente
 */
	void treina_sistema_pacote(sistema_neural_t & S, pares_x_y const & dados, NUM_T const taxa_aprendizado, NUM_T const lambda);
	void treina_sistema_epoca(sistema_neural_t & S, classes_separadas_t const & treino,	int const N_por_pacote,	NUM_T const eta, NUM_T const lambda);

//gera vetor x representativo da imagem fornecida
	vec_f gera_x_de_bitmap(bm_ref_t const & img);
	
//converte y de saída para índice de classe 
	int mapa_y_classe(vec_f const & y);

//gera um pacote_x_y de N amostras, começando em i0, nos dados fornecidos
	pares_x_y gera_pacote(classes_separadas_t const& dados, int i0, int N);

//função de teste
	struct teste_resultado_t
	{
		int N_tot;
		int N_acertos_tot;
		int N_classes;
		vec_i N_amostras_classe;
		vec_i N_acertos_classe;
	};
	teste_resultado_t testa_sistema(sistema_neural_t const & S, classes_separadas_t const & teste);
	
//função de ativação, sigmóide 
	NUM_T sigmoide(NUM_T x);
	vec_f sigmoide(vec_f x);
	
//função de ativação, softmax 
	vec_f softmax(vec_f x);
	
//derivada da função de ativação
	NUM_T dsig_dx(NUM_T x);
	vec_f dsig_dx(vec_f x);
	vec_f dsoftmaxj_dx(vec_f x, int j);
	
//função de custo
	NUM_T custo_softmax_log(vec_f const & y_ref, vec_f const & y_app);
	
//multiplicação de Hadamard/Schur/por elemento
	vec_f Hadamard(vec_f const & a, vec_f const & b);

#endif // fim de header guard SISNEU_INCLUSO