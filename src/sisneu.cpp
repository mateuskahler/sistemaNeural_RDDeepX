
#include "sisneu.hpp"

#include <cmath>
#include <set>
#include <map>
#include <random>

#include <fstream>
#include <sstream>
#include <iomanip>

//gerador aleatório estático global
static std::mt19937_64 RGEN;

//////////////////////////////////////////////////////
//////////////////////////////////////////////////////
///////    Matriz_t           ////////////////////////
//////////////////////////////////////////////////////

int Matriz_t::Largura()const
{
	if(mem.empty())
		{return 0;}
	return static_cast<int>(mem[0].size()); //todas linhas têm mesmo tamanho, então a largura é calculada pelo tamanho da primeira linha
}

int Matriz_t::Altura()const
{
	return static_cast<int>(mem.size()); //altura == número de linhas
}

//altera as dimensões e zera novos elementos
void Matriz_t::redimensiona(int largura, int altura)
{
	if(largura<=0 || altura<=0)
		{ mem.clear(); return;}
	bool altura_mudou = false; //flag que indica mudança de altura (e força ajuste para manter todas as linhas de mesmo tamanho)
	if(altura != static_cast<int>(mem.size()))
	{
		altura_mudou = true; 
		mem.resize(altura); 
	}
	//como a matriz é, de fato, um vetor de vetores, as linhas tem de ser reservadas uma a uma
	if((largura != static_cast<int>(mem[0].size())) || (altura_mudou))
	{
		for(auto & l:mem)
			{l.resize(largura, static_cast<NUM_T>(0));}
	}
}

//zera todos os elementos
void Matriz_t::zera()
{
	for(auto & l : mem)
	{
		for(auto & c : l)
			{ c = static_cast<NUM_T>(0); }
	}
}

//////////////////////////////////////////////////////
//////////////////////////////////////////////////////
///////    camada_neural_t    ////////////////////////
//////////////////////////////////////////////////////

void camada_neural_t::constroi(int N_saidas, int N_entradas)
{
	w.redimensiona(N_entradas, N_saidas);
	w.zera();
	b = vec_f(N_saidas, static_cast<NUM_T>(0));
}

// z = w*x + b
vec_f camada_neural_t::resolve_z(vec_f const & x)const
{
	#ifndef NDEBUG
		if((static_cast<int>(x.size())!=w.Largura()) || (x.size()==0))
				{throw std::logic_error("camada_neural_t::resolve(i), tamanho de vetores incompativel");}
	#endif
	int const La = w.Altura();
	int const Le = w.Largura();
	// bias e ponderação de todas entradas 
		vec_f z_buf(b);
		for(int j=0; j<La; j++)
		{
			for(int k=0; k<Le; k++)
				{z_buf[j] += x[k] * w.mem[j][k];}
		}
	return z_buf;
}

int camada_neural_t::n_entradas()const
{
	return w.Largura();
}

int camada_neural_t::n_saidas()const
{
	return w.Altura();
}

//////////////////////////////////////////////////////
//////////////////////////////////////////////////////
///////    gradiente_camada_neural_t    //////////////
//////////////////////////////////////////////////////

gradiente_camada_neural_t & gradiente_camada_neural_t::operator += (gradiente_camada_neural_t const & dG)
{
	int const A = dCdw.Altura();
	int const L = dCdw.Largura();
	for(int j=0; j<A; j++)
	{
		for(int k=0; k<L; k++)
		{
			dCdw.mem[j][k] += dG.dCdw.mem[j][k];
		} 
		dCdz[j] += dG.dCdz[j];
	}
	return (*this);
}

gradiente_camada_neural_t & gradiente_camada_neural_t::operator *= (NUM_T const k)
{
	for(auto & wl:dCdw.mem)
	{
		for(auto & wc:wl)
			{ wc *= k; } 
	}
	for(auto & z:dCdz)
		{ z *= k; }
	return (*this);	
}

//////////////////////////////////////////////////////
//////////////////////////////////////////////////////
///////    gradientes_t       ////////////////////////
//////////////////////////////////////////////////////

gradientes_t & gradientes_t::operator += (gradientes_t const & dG)
{
	int const nC = gradientes.size();
	for(int c=0; c< nC; c++)
		{ gradientes[c] += dG.gradientes[c]; }
	return (*this);	
}

gradientes_t & gradientes_t::operator *= (NUM_T const k)
{
	int const nC = gradientes.size();
	for(int c=0; c< nC; c++)
		{ gradientes[c] *= k; }
	return (*this);	
}

void gradientes_t::zera()
{
	int const nC = gradientes.size();
	for(int c=0; c< nC; c++)
	{
		gradientes[c].dCdw.zera();
		for(auto & b: gradientes[c].dCdz)
			{ b = 0;}
	}
}
		
//////////////////////////////////////////////////////
//////////////////////////////////////////////////////
///////    sistema_neural_t   ////////////////////////
//////////////////////////////////////////////////////

void sistema_neural_t::dimensiona(int const eN, int const cN, vec_i const cnN)
{
	if((cN != static_cast<int>(cnN.size())))
			{throw std::logic_error("sistema_neural_t::dimensiona: quantidade de camadas incompativel");}
	if(cN < 2)
			{throw std::logic_error("sistema_neural_t::dimensiona: quantidade de camadas incompativel");}
	if( (cN<=0) || (eN<=0) )
			{throw std::logic_error("sistema_neural_t::dimensiona: configuracao incompativel");}
	for(auto nc : cnN)
	{
		if(nc<=0)
			{throw std::logic_error("sistema_neural_t::dimensiona: camada sem neuronios");}
	}
	camadas.resize(cN);
	camadas[0].constroi(cnN[0], eN);
	for(int c=1; c<cN; c++)
	{
		camadas[c].constroi(cnN[c], cnN[c-1]);
	}
}

/*
 * o sorteio de coeficientes visa quebrar simetria inicial do sistema
 * referencias consultada indica que distribuição normal dos pesos em terno de zero oferece bom comportamento inicial
 * os pesos são normalizados pelo número de entradas, tentando evitar pesos muito proeminentes no estado inicial
 * */
void sistema_neural_t::sorteia_coefs()
{
	std::normal_distribution<NUM_T> dist_n_dev1(0.f, 1.f);
	for(unsigned c=0; c<camadas.size(); c++)
	{
		for(int e=0; e<camadas[c].w.Altura(); e++)
		{
			NUM_T const am_escala = std::sqrt(2.)/camadas[c].w.Largura();
			for(int k=0; k<camadas[c].w.Largura(); k++)
			{
				camadas[c].w.mem[e][k] = (dist_n_dev1(RGEN))*am_escala;
			}
			camadas[c].b[e] = 0.;
		}
	}
}

vec_f sistema_neural_t::resolve(vec_f const & x)const
{
	int const NC = camadas.size();
	if(NC == 0)
		{return vec_f();}
	//propaga salvando intermediários
		vec_f buf_a;
		for(int c=0; c<NC; c++)
		{
			//a entrada da primeira camada é x
				vec_f const & e =  (c>0) ? buf_a : x;
				vec_f buf_z  = camadas[c].resolve_z(e);
			//a ativação da última camada é softmax, das intermediárias é sigmoide
				if(c==(NC-1))
				{
					buf_a = softmax(buf_z);
				}
				else
				{
					buf_a = sigmoide(buf_z);
				}
		}
	return buf_a;
}

//backpropagation incremental
void sistema_neural_t::back_prop_inc(vec_f const & x, vec_f const & y_ref, gradientes_t & dest)const
{
	//lambda para gradiente dos pesos em w
		auto dC_dw_calc_inc = [](
				Matriz_t & w,
				vec_f const & entrada,
				vec_f const & dC_dz ) -> void
			{
				int const JN = dC_dz.size();  //# saídas   -> linhas de w
				int const KN = entrada.size();//# entradas -> colunas de w
				for(int j=0; j<JN; j++)
				{
					for(int k=0; k<KN; k++)
						{	w.mem[j][k] += entrada[k]*dC_dz[j];	}
				}
				return;
			};
	//lambda para gradiente das ativações em a
		auto calc_dC_da = [](
				vec_f & b,
				Matriz_t const & w_posterior,
				vec_f const & dC_dz_posterior ) -> void
			{
				int const JN = w_posterior.Largura(); // # de entradas posterior = número de saídas atual = linhas de w
				int const HN = w_posterior.Altura();  // # de saídas posterior
				for(int j=0; j<JN; j++)
				{
					b[j] = w_posterior.mem[0][j] * dC_dz_posterior[0];
					for(int h=1; h<HN; h++)						
						{	b[j] += w_posterior.mem[h][j] * dC_dz_posterior[h];	}
				}
				return;
			};
	int const NC = camadas.size();
	if(NC == 0)
		{return;}
	//propaga salvando intermediários 
	//  (esses vetores de vetores são thread local para manter memória alocada entre treinos, como se estáticos,
	//    mas estáticos por thread, para não concorrerem memória caso treinos nasçam de threads distos)
		thread_local std::vector<vec_f> buf_a;
		thread_local std::vector<vec_f> buf_da_dz;
		thread_local std::vector<vec_f> buf_dC_da;
		thread_local std::vector<vec_f> buf_dC_dz;
		buf_a.resize(NC);
		buf_da_dz.resize(NC-1);
		buf_dC_da.resize(NC-1);
		buf_dC_dz.resize(NC);
		for(int c=0; c<NC; c++)
		{
			//a entrada da primeira camada é x
				vec_f const & e =  (c>0) ? buf_a[c-1] : x;
				vec_f buf_z  = camadas[c].resolve_z(e);
			//a ativação da última camada é softmax, das intermediárias é sigmoide
				if(c==(NC-1))
				{
					buf_a[c] = softmax(buf_z);
					int const na = dest.gradientes[c].dCdz.size();
					//selecioan índice de classe correta
						int R = 0;
							for(int j=0; j<na; j++)
							{ 
								if(y_ref[j]>0.5)
									{R=j; break;}
							}
					//derivada parcial de termo de softmax da saída correta em relação a z
						NUM_T const iaR = (-1./buf_a[c][R]);
						auto daRdz = dsoftmaxj_dx(buf_z, R);
						buf_dC_dz[c].clear();
						for(auto d:daRdz)
							{ buf_dC_dz[c].emplace_back(iaR*d); }
					//acrescenta gradiente do bias
							for(int j=0; j<na; j++)
							{ 
								dest.gradientes[c].dCdz[j] += buf_dC_dz[c][j];
							}
					//acrescenta gradiente dos pesos
							int const ne = dest.gradientes[c].dCdw.Largura();
							for(int j=0; j<na; j++)
							{
								for(int k=0; k<ne; k++)
								{
									dest.gradientes[c].dCdw.mem[j][k] += buf_dC_dz[c][j]*buf_a[c-1][k];
								}
							}
				} // fim do tratamento da camada softmax
				else
				{
					buf_a[c] = sigmoide(buf_z);	
					buf_da_dz[c] = dsig_dx(buf_z);
				}
		}
	//back-propaga gradientes por camadas de sigmoides
		for(int c=NC-2; c>=0; c--)
		{
			//calcula dC/da
				int na = dest.gradientes[c].dCdz.size();
				buf_dC_da[c].resize(na);
				calc_dC_da(buf_dC_da[c], camadas[c+1].w, buf_dC_dz[c+1]);
			//calcula dC/dz e incrementa destino
			{
				buf_dC_dz[c] = Hadamard(buf_dC_da[c], buf_da_dz[c]);
				int na = dest.gradientes[c].dCdz.size();
				for(int k=0; k<na; k++)
					{ dest.gradientes[c].dCdz[k] += buf_dC_dz[c][k]; }
			}
			//calcula dC/dw e incrementa destino
			{
				if(c>0)
				{
					dC_dw_calc_inc(dest.gradientes[c].dCdw, buf_a[c-1], buf_dC_dz[c]);
				}
				else //na primeira camada, a entrada anterior é x, e não a saída de outra camada
				{
					dC_dw_calc_inc(dest.gradientes[c].dCdw, x, buf_dC_dz[c]);
				}
			}
		}
	return;
}

sistema_neural_t & sistema_neural_t::deriva_por(gradientes_t const & dS)
{	
	//validação de dimensões removida em release
	#ifndef NDEBUG
		if((dS.gradientes.size() != camadas.size()) || (camadas.size()==0))
			{throw std::logic_error("sistema_neural_t::deriva_por(dS) : quantidade de camadas incompativel");}
		for(unsigned int k=0; k<camadas.size();k++)
		{
			if( (dS.gradientes[k].dCdw.Largura() != camadas[k].w.Largura()) || (camadas[k].w.Largura()==0) || 
					(dS.gradientes[k].dCdw.Altura()  != camadas[k].w.Altura())  || (camadas[k].w.Altura() ==0) )
				{throw std::logic_error("sistema_neural_t::deriva_por(dS) : quantidade de neuronios incompativel");}
		}
	#endif
	//adiciona termo a termo
		for(unsigned int c=0; c<camadas.size(); c++)
		{
			for(int j=0; j<camadas[c].w.Altura(); j++)
			{
				for(int k=0; k<camadas[c].w.Largura(); k++)
				{
					//pesos
					camadas[c].w.mem[j][k] += dS.gradientes[c].dCdw.mem[j][k];
				}
				//bias
				camadas[c].b[j] += dS.gradientes[c].dCdz[j];
			}
		}
	return (*this);
}

gradientes_t sistema_neural_t::gera_gradiente_nulo()const
{
	int const cN = camadas.size();
	gradientes_t G0;
	G0.gradientes.resize(cN);
	for(int c=0; c<cN; c++)
	{
		int const ne = camadas[c].n_entradas();
		int const ns = camadas[c].n_saidas();
		G0.gradientes[c].dCdw.redimensiona(ne, ns);
		G0.gradientes[c].dCdz.resize(ns, 0);
	}
	return G0;
}

bool sistema_neural_t::salva_para_arquivo(std::string const arq_caminho)const
{
	if(camadas.empty())
		{return false;}
	std::ofstream arq(arq_caminho, std::ios::out | std::ios::trunc);
	if(!arq)
		{return false;}
	//lambda para escrever ponto flutuante em notação científica com sete casas de precisão
		auto salva_f_ = [&](NUM_T v)
			{	arq << std::setw(15) << std::scientific << std::setprecision(7) << v << ' ';	};
	//escreve linha de comentário inicial
		arq << " =) arquivo de pesos para sistema neural\n";
	//escreve linha com número de camadas
		arq << camadas[0].n_entradas() << " entradas\n";
		arq << camadas.size() << " camadas, com\n";
	//escreve linha com número de neurônios por camada
		for(auto & c : camadas)
			{ arq << c.n_saidas() << " ";}
		arq << "neuronios respectivamente\n";
	//salva coeficientes
		for(auto const & c : camadas)
		{
			int const nE = c.n_entradas();
			int const nS = c.n_saidas();
			//salva b
				for(int j=0; j<nS; j++)
					{	salva_f_(c.b[j]);	}
				arq << '\n';
				arq.flush();
			//salva w
				for(int j=0; j<nS; j++)
				{
					for(int k=0; k<nE; k++)
						{	salva_f_(c.w.mem[j][k]);	}
					arq << '\n';
					arq.flush();
				}
		}
	return (bool)arq;
	}

bool sistema_neural_t::carrega_de_arquivo(std::string const arq_caminho)
{
	std::ifstream arq(arq_caminho);
	//retorna 'false' silenciosamente se falhar abrir arquivo
		if(!arq)
			{return false;}
	int BUF_N = 512; //buffer de 512 bytes, suficiente para primeiras linhas
	std::vector<char> buf(BUF_N);
	arq.getline(&(buf[0]), 512, '\n'); //linha de comentário no arquivo
	// lê linha com número de entradas
	int NE = 0;
		{
			arq.getline(&(buf[0]), 512, '\n'); 
			std::stringstream ss(&(buf[0]));
			ss >> NE;
			if((!arq) || (NE<=0))
				{return false;}
		}
	// lê linha com número de camadas
	int NC = 0;
		{
			arq.getline(&(buf[0]), 512, '\n');
			std::stringstream ss(&(buf[0]));
			ss >> NC;
			if((!arq) || (NC<=0))
				{return false;}
		}
	// linha com número de neurônios por camada
	vec_i NEs(NC);
		{
			arq.getline(&(buf[0]), 512, '\n'); 
				std::stringstream ss(&(buf[0]));
			for(int k=0; k<NC; k++)
			{
				int nn = 0;
				ss >> nn;
				if((!arq) || (nn<=0))
					{return false;}
				NEs[k] = nn;
			}
		}
	sistema_neural_t S;
	S.dimensiona(NE, NC, NEs);
	for(int c=0; c<NC; c++)
	{
		int const nE = S.camadas[c].n_entradas();
		int const nS = S.camadas[c].n_saidas();
		//lê b
			for(int j=0; j<nS; j++)
				{ arq >> S.camadas[c].b[j]; }
		//lê w
			for(int j=0; j<nS; j++)
			{
				for(int k=0; k<nE; k++)
					{	arq >> S.camadas[c].w.mem[j][k];	}
			}
	}
	bool ok = (bool)arq;
	if(ok)
	{
		(*this) = S;
	}
	return ok;
}
//////////////////////////////////////////////////////
//////////////////////////////////////////////////////
///////    classes_separadas_t   /////////////////////
//////////////////////////////////////////////////////

classes_separadas_t::classes_separadas_t(dig_refs_t refs)
	: mem(std::move(refs))
{
	//reordena entrada (isso evita vício do sistema caso entrada esteja ordenada)
	int const NA = mem.size();
	std::uniform_int_distribution<int> dis(0, NA-1);
		for(int k=0; k<NA; k++)
		{
			int ni = dis(RGEN);
			std::swap(mem[k], mem[ni]);
		}
	//separa as referências em classes por etiqueta
	//  dois passos, para manter classes ordenadas crescentes por etiqueta original
	std::map<int8_t, int> mp_e_i; //mapa de etiquetas para índice de classe
		{
			std::set<int8_t> conj_e; //conjunto de etiquetas
			for(auto const & r : mem)
				{ conj_e.insert(r.etiqueta); }
			int const NC = conj_e.size(); //número de etiquetas únicas
			inds_classificados.resize(NC);
			//para cada etiqueta única, gera uma nova classe interna
				auto etiqueta_it = conj_e.begin();
				for(int c=0; c<NC; c++) //como o std::set é ordenado crescente, isso garante que os índices internos de classe também o serão
				{
					int8_t const r_e = *(etiqueta_it);
					etiqueta_it++;
					mp_e_i[r_e] = c;
					inds_classificados[c].etiqueta_original = r_e;
					inds_classificados[c].y_ref = vec_f(NC, static_cast<NUM_T>(0));
					inds_classificados[c].y_ref[c] = static_cast<NUM_T>(1);
				}
		}
	//salva índice de cada item, de vetor contínuo para lista de amostras da respectiva classe
	for(int k=0; k<NA; k++)
	{
		int8_t const r_e = mem[k].etiqueta;
		mem[k].etiqueta = mp_e_i[r_e];
		inds_classificados[mem[k].etiqueta].amostras_i.emplace_back(k);
	}
}

int classes_separadas_t::quantidade_classes()const
{
	return inds_classificados.size();
}

int classes_separadas_t::quantidade_entradas()const
{
	return 28*28;
}

int classes_separadas_t::amostras_total()const
{
	return static_cast<int>(mem.size());
}

int classes_separadas_t::amostras_classe(int i)const
{
	//validação de dimensões removida em release
	#ifndef NDEBUG
		if((i>=static_cast<int>(inds_classificados.size())) || (i<0))
			{throw std::out_of_range("classes_separadas_t::amostras_classe : classe inexistente");}
	#endif
	return inds_classificados[i].amostras_i.size();
}

int classes_separadas_t::gera_indice(int classei, int amostrai)const
{
	return inds_classificados[classei].amostras_i[amostrai];
}

vec_f classes_separadas_t::gera_x(int indice)const
{
	return gera_x_de_bitmap(mem[indice].img);
}

vec_f classes_separadas_t::gera_y(int indice)const
{
	return inds_classificados[mem[indice].etiqueta].y_ref;
}

par_x_y_t classes_separadas_t::gera_xy(int indice)const
{
	return par_x_y_t{gera_x(indice), gera_y(indice)};
}

void classes_separadas_t::exclui_classe(int classe_id)
{
	auto teste_etiqueta = [classe_id](dig_ref_t const & d){return (d.etiqueta == classe_id);};
	mem.erase(std::remove_if(mem.begin(), mem.end(), teste_etiqueta), mem.end());
	for(auto & r: mem)
		{r.etiqueta = inds_classificados[r.etiqueta].etiqueta_original;}
	(*this) = classes_separadas_t(std::move(mem));
}

//////////////////////////////////////////////////////
//////////////////////////////////////////////////////
///////    funções auxiliares    /////////////////////
//////////////////////////////////////////////////////

/**
 *  treina uma época (usa todas amostras do banco de treino fornecido)
 *  S : sistema a ser treinado
 *  treino : amostras usadas para treino
 *  N_por_pacote : tamanho de 'mini-batch''
 *  eta : taxa de aprendizado
 *  lambda : fator de normalização L2
 */
void treina_sistema_epoca( 
	sistema_neural_t & S, classes_separadas_t const & treino,
	int const N_por_pacote,
	NUM_T const eta, NUM_T const lambda)
{
	//calcula número de pacotes
		int const NA = treino.amostras_total();
		int const N_pacotes = NA/N_por_pacote;
		int a_resto         = NA%N_por_pacote; //amostras restantes de divisão exata
		int pos = 0; //posição linear no banco de amostras
	//treina sistema usando cada pacote
		for(int k=0; k<N_pacotes; k++)
		{
			int N_por_pacote_esc = N_por_pacote;
			//esse pequeno ajuste garante que todo o banco seja usado, adicionando uma amostra a alguns testes quando a divisão não for perfeita
				if((k>a_resto) && (a_resto>0))
					{ N_por_pacote_esc++;
						a_resto--; }
			treina_sistema_pacote(S, gera_pacote(treino, pos, N_por_pacote_esc), eta, lambda);
			pos += N_por_pacote_esc;
		}
}

/**
 *  treina um pacote
 *  S : sistema a ser treinado
 *  pares_x_y : amostras usadas para treino
 *  taxa_aprendizado : taxa de aprendizado
 *  lambda : fator de normalização L2
 */
void treina_sistema_pacote(sistema_neural_t & S, pares_x_y const & dados, NUM_T const taxa_aprendizado, NUM_T const lambda)
{	
	//mantém memória thread local suficiente reservada para gradientes
		thread_local gradientes_t Gp;
		int const NC = S.camadas.size();
		Gp.gradientes.resize(NC);
		for(int c=0; c<NC; c++)
		{	
			Gp.gradientes[c].dCdw.redimensiona(S.camadas[c].n_entradas(), S.camadas[c].n_saidas());
			Gp.gradientes[c].dCdz.resize(S.camadas[c].n_saidas());
		}
		Gp.zera();
	//acumula todos gradientes do pacote
		for(auto const & d:dados)
			{	S.back_prop_inc(d.x, d.y, Gp);	}
	//gradiente médio vezes taxa de aprendizado e normalização L2
		NUM_T const escala = (taxa_aprendizado/static_cast<NUM_T>(dados.size()));
			for(int c=0; c< NC; c++)
			{
				int const Na = S.camadas[c].n_saidas();
				for(int j=0; j<Na; j++)
				{
					int const Ne = S.camadas[c].n_entradas();
					for(int k=0; k<Ne; k++)
					{
						auto & g = Gp.gradientes[c].dCdw.mem[j][k];
						auto const & s = S.camadas[c].w.mem[j][k];
						g = -(escala*g + lambda*s);
					}
					auto & g = Gp.gradientes[c].dCdz[j];
					auto const & s = S.camadas[c].b[j];
					g = -(escala*g + lambda*s);
				}
			}
	S.deriva_por(Gp); //aplica
	return;
}

int mapa_y_classe(vec_f const & y)
{
	int mi = 0;
	if(y.empty())
		{return 0;}
	int ni = y.size();
	for(int k=1; k<ni; k++)
	{
		if(y[k] > y[mi])
			{ mi = k; }
	}
	return mi;
}

bool mesmo_mapa(vec_f ya, vec_f yb)
{
	if(ya.empty() || yb.empty())
		{return false;}
	return (mapa_y_classe(ya) == mapa_y_classe(yb));
}

//gera vetor x representativo da imagem fornecida
vec_f gera_x_de_bitmap(bm_ref_t const & img)
{
	NUM_T constexpr invFF = 1.f/255.f;
	//escala para valor unitário e coleta média
		vec_f x(28*28);
		NUM_T media(0);
		for(int yi=0; yi<28; yi++)
		{
			for(int xi=0; xi<28; xi++)
			{
				NUM_T v = static_cast<NUM_T>(img[yi][xi])*invFF;
				media += v;
				x[yi*28+xi] = v;
			}
		}
	//reduz média de cada componente
	media /= (28*28);
		for(int yi=0; yi<28; yi++)
		{
			for(int xi=0; xi<28; xi++)
				{	x[yi*28+xi] -= media;	}
		}
	return x;
}

//gera um pacote_x_y de N amostras, começando em i0, nos dados fornecidos
//o pacote pode ser usado para treino do sistema neural
pares_x_y gera_pacote(classes_separadas_t const& dados, int i0, int N)
{
	pares_x_y R;
	int const i_max = dados.amostras_total();
	for(int k=0; k<N; k++)
	{
		int i = (i0+k)%i_max;
		R.push_back(dados.gera_xy(i));
	}
	return R;
}

teste_resultado_t testa_sistema(sistema_neural_t const & S, classes_separadas_t const & teste)
{
	teste_resultado_t R;
	R.N_tot = teste.amostras_total();
	R.N_acertos_tot = 0;
	R.N_classes = teste.quantidade_classes();
	R.N_acertos_classe.resize(R.N_classes, 0);
	R.N_amostras_classe.resize(R.N_classes, 0);
	for(int k=0; k<R.N_tot; k++)
	{
		auto x = teste.gera_x(k);
		auto y_ref = teste.gera_y(k);
		auto y_app = S.resolve(x);
		R.N_amostras_classe[mapa_y_classe(y_ref)]++;
		if(mesmo_mapa(y_ref, y_app))
		{
			R.N_acertos_tot++;
			R.N_acertos_classe[mapa_y_classe(y_ref)]++;
		}
	}
	return R;
}

NUM_T sigmoide(NUM_T x)
{
	//limitada
	NUM_T constexpr xmin = -15.f;
	NUM_T constexpr xmax = +15.f;
		if(x < xmin)
			{ x = xmin ; }
		else if(x > xmax)
			{ x = xmax ; }
	NUM_T const xdn = -x;
	NUM_T ee = std::exp(xdn);
	return static_cast<NUM_T>(1./(1. + ee));
}

vec_f sigmoide(vec_f x)
{
	// vetor de entrada -sigmoide-> vetor de saída
	int const N = x.size();
	for(int k=0; k<N; k++)
		{ x[k] = (sigmoide(x[k])); }
	return x;	
}

vec_f softmax(vec_f x)
{
	NUM_T soma(0);
	for(auto const xi:x)
		{ soma += std::exp(xi);}
	for(auto & xi:x)
		{ xi = std::exp(xi)/soma;}
	return x;
}
	
NUM_T dsig_dx(NUM_T x)
{
	NUM_T s = sigmoide(x);
	return (s*(1.f-s));
}

vec_f dsig_dx(vec_f x)
{
	// vetor de entrada -dsig_dx-> vetor de saída
	int const N = x.size();
	for(int k=0; k<N; k++)
		{ x[k] = (dsig_dx(x[k])); }
	return x;	
}

// a derivada da softmax é melhor explicada no relatório técnico.
// em suma, a função é observada no termo de saída desejado em relação a cada termo da entrada,
// e a forma apropriada foi derivada analiticamente e programada nesta função
vec_f dsoftmaxj_dx(vec_f x, int j)
{
	int const N(x.size());
	NUM_T const ex = std::exp(x[j]); // exp( x[j] )
	NUM_T H(0); // soma de todos outros termos exponenciados
		for(int k=0; k<N; k++)
		{
			if(k != j)
			 { H += std::exp(x[k]); }
		}
	//cálcula da derivada parcial de saída [j] de softmax em relação a cada termo
	NUM_T const Q = (((ex*ex) + (2*H*ex)) + H*H);
	NUM_T const eiQ = ex/Q;
	for(int k=0; k<N; k++)
		{
			if(k == j)
			{
				x[j] = H*eiQ;
			}
			else
			{
				x[k] = -(std::exp(x[k])*eiQ) ;
			}
		}
	return x;
}


NUM_T custo_softmax_log(vec_f const & y_ref, vec_f const & app)
{
	int const n = y_ref.size();
	//y_ref contém apenas uma classe correta, de peso 1
	for(int k=0; k<n; k++)
	{
		if(y_ref[k]>0.5)
			{	return -std::log(app[k]);	}
	}
	return 0;
}

// multiplicação de vetores termo a termo
vec_f Hadamard(vec_f const & a, vec_f const & b)
{	
	//validação de dimensões removida em release
	#ifndef NDEBUG
		if((a.size()!=b.size()) || (a.size()==0))
			{throw std::out_of_range("Hadamard(a,b) : dimensoes invalidas");}
	#endif
	// transform: vetores de entrada -multiplicação-> vetor de saída
	int const N = a.size();
	vec_f h;
	h.reserve(N);
	for(int k=0; k<N; k++)
		{		h.emplace_back(a[k]*b[k]);	}
	return h;	
}