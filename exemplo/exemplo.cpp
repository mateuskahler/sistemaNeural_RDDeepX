/**
 * 
 *   Demonstração do uso da rede neural treinada para identificar caracteres escritos a mão.
 * 
 *   Ao ser executado, o programa carrega imagens e interpreta cada caractere, imprimindo-os no console.
 * 
 *   Junto com este arquivo são fornecidas as imagens desafio1 e desafio2,
 * que representam o texto a ser interpretado.
 *   Também são fornecidos os arquivos sistema_treinado_letras e sistema_treinado_numeros,
 * quem contém os coeficientes das redes pré-treinadas.
 * 
 * Este programa e biblioteca associada foram criados para resolver um dos desafios 
 * de pesquisa e desenvolvimento propostos pela DeepX 
 * (exceto pelo leitor de arquivos bmp, que adaptei de um código antigo).
 * 
 * Para compilar:
 * 
 *  g++ -o exemplo exemplo.cpp ../src/sisneu.cpp -I../src 
 * 
 *   É necessário fornecer caminho para incluir "sisneu.hpp" e linkar com "sisneu.cpp".
 * A linha acima mostra exemplo disto no GCC, com a estrutura de pastas fornecidas no repositório.
 *
 * Mateus Kahler, Setembro de 2018
 * */
 
// inclui classes desenvolvidas para redes neurais
	#include "sisneu.hpp"

//bibliotecas padrão requeridas
	#include <string>
	#include <iostream>
	#include <iomanip>
	#include <fstream>
	#include <cstdint>
	
//define tipo que representa vetor de imagens (um vetor de arrays de 28*28 pixels)
	typedef std::vector<bm_ref_t> bms_t;
	
//função que carrega bmp e divide em grades de caracteres
	bms_t CarregaArquivoBMP(std::string nome_arquivo);

//função que recebe sistema neural e imagem, e interpreta letra
	char identificar_letra ( sistema_neural_t const & S, bm_ref_t img );

//função que recebe sistema neural e imagem, e interpreta número
	char identificar_numero ( sistema_neural_t const & S, bm_ref_t img );

//mapa de etiquetas para caracteres numéricos, segundo documentação do EMNIST
	std::vector<std::pair<uint8_t, char>> const
		mapa_EMNIST_treinado_numeros { 
			{  0, '0' },		{  1, '1' },		{  2, '2' },		{  3, '3' },		{  4, '4' },
			{  5, '5' },		{  6, '6' },		{  7, '7' },		{  8, '8' },		{  9, '9' } };

//mapa de etiquetas para letras, segundo documentação do EMNIST
	std::vector<std::pair<uint8_t, char>> const
		mapa_EMNIST_treinado_letras { 
			{  0, 'A' },		{  1, 'B' },		{  2, 'C' },		{  3, 'D' },
			{  4, 'E' },		{  5, 'F' },		{  6, 'G' },		{  7, 'H' },
			{  8, 'I' },		{  9, 'J' },		{ 10, 'K' },		{ 11, 'L' },
			{ 12, 'M' },		{ 13, 'N' },		{ 14, 'O' },		{ 15, 'P' },
			{ 16, 'Q' },		{ 17, 'R' },		{ 18, 'S' },		{ 19, 'T' },
			{ 20, 'U' },		{ 21, 'V' },		{ 22, 'W' },		{ 23, 'X' },
			{ 24, 'Y' },		{ 25, 'Z' },		{ 26, 'a' },		{ 27, 'b' },
			{ 28, 'd' },		{ 29, 'e' },		{ 30, 'f' },		{ 31, 'g' },
			{ 32, 'h' },		{ 33, 'n' },		{ 34, 'q' },		{ 35, 'r' },		{ 36, 't' } };

//ponto de entrada
int main(int argc, char **argv)
{
	////////////////////////////////////////////////////////////////
	/// primeiro desafio: exemplo de uso da rede treinada pra identificar letras
	//carrega rede neural treinada para letras
		sistema_neural_t S;
		if(!S.carrega_de_arquivo("sistema_treinado_letras"))
		{ 
			std::cout << "erro ao carregar \'sistema_treinado_letras\'" << std::endl;
			std::cout << "o arquivo esta na pasta do executavel?" << std::endl;
			return 1;
		}
	//carrega imagem com frase provocativa
		auto cs = CarregaArquivoBMP("desafio1.bmp");
	//interpreta cada caractere da imagem
		std::string resultado;
		for(auto c:cs)
			{	resultado += identificar_letra(S, c);	}
		//imprime resultado no console
		std::cout << std::endl << '\t' << resultado << std::endl << std::endl;
	
	////////////////////////////////////////////////////////////////
	/// segundo desafio: exemplo de uso da rede treinada pra identificar números
	//carrega rede neural treinada para números
		if(!S.carrega_de_arquivo("sistema_treinado_numeros"))
		{ 
			std::cout << "erro ao carregar \'sistema_treinado_numeros\'" << std::endl;
			std::cout << "o arquivo esta na pasta do executavel?" << std::endl;
			return 1;
		}
		
	//carrega imagem com dígitos
		cs = CarregaArquivoBMP("desafio2.bmp");
		resultado = std::string("");
	//interpreta cada caractere da imagem
		for(auto c:cs)
			{	resultado += identificar_numero(S, c);	}
		//imprime resultado no console
		std::cout << std::endl << '\t' << resultado << std::endl << std::endl;
	
	return 0;
}

/**********************************************************
 *  implementação das funções auxilires....
 * ***********************************************           */
 /**
 * CarregaArquivoBMP : função que carrega um arquivo bmp e divide em partes de 28*28 pixels de 1 byte
 * 
 * argumentos :
 * 	nome_arquivo : caminho para o arquivo a ser carregado
 *
 * a função imprime informações de falha para o console caso não encontre/compreenda o arquivo
 * 
 */
bms_t CarregaArquivoBMP(std::string nome_arquivo)
{
  using namespace std;
  //  Existem vários padrões para arquivos bitmap, todos começam com 
  //um cabeçalho obrigatório  indicando o tipo de arquivo e tamanho, 
  //seguido por um dos modelos de cabeçalho com informações de 
  //resolução e esquema de cores, e possíveis cabeçalhos adicionais 
  //com esquemas de compressão ou identificando formatos exóticos.
  //
  //  Num equilíbrio entre simplicidade e portabilidade, optei pelo 
  //cabeçalho BITMAPINFOHEADER, o mais difundido e usado por padrão 
  //ao salvar imagens no MSPaintBrush.
  struct bitmap_header_t
  {
    char formato[2];
    uint32_t tamanho_total;
    uint32_t programa;
    uint32_t offset_dados;
    uint32_t tamanho_info_header;
    int32_t largura;
    int32_t altura;
    uint16_t planos;
    uint16_t bits_por_pixel;
    uint32_t codigo_compressao;
    uint32_t tamanho_imagem;
    int32_t res_hor;
    int32_t res_ver;
    uint32_t cores_paleta;
    uint32_t cores_importantes;
  } arq_info; // cabeçalho do arquivo
  bms_t arq_imagem; //resultado
  ifstream arq_stream(nome_arquivo, std::ios::binary); //abre arquivo como stream de dados
  if(!arq_stream) //se falhar a abertura do arquivo
  {
    //percebi que costumo esquecer a extensão do arquivo durante os testes,
    //então adicionei esse teste que adiciona a extensão .bmp e tenta recarregar
    //o arquivo.
    std::string::size_type n = nome_arquivo.find(".");
    if(n == std::string::npos)
    {
      nome_arquivo += ".bmp";
      arq_imagem = CarregaArquivoBMP(nome_arquivo);
    }
    else
    {
      std::cout << "Erro ao tentar abrir arquivo " << nome_arquivo << std::endl;
      std::cout << "(A imagem esta na mesma pasta que o executavel?)";
      std::cout << std::endl << std::endl;
    }
  }
  else
  {
    std::cout << "Carregando arquivo " << nome_arquivo << "..." << std::endl;
    //lê na ordem, de acordo com especificação BITMAPINFOHEADER...
			arq_stream.read(arq_info.formato, 2);
			arq_stream.read(reinterpret_cast<char*>(&arq_info.tamanho_total), 4);
			arq_stream.read(reinterpret_cast<char*>(&arq_info.programa), 4);
			arq_stream.read(reinterpret_cast<char*>(&arq_info.offset_dados), 4);
			arq_stream.read(reinterpret_cast<char*>(&arq_info.tamanho_info_header), 4);
			arq_stream.read(reinterpret_cast<char*>(&arq_info.largura), 4);
			arq_stream.read(reinterpret_cast<char*>(&arq_info.altura), 4);
			arq_stream.read(reinterpret_cast<char*>(&arq_info.planos), 2);
			arq_stream.read(reinterpret_cast<char*>(&arq_info.bits_por_pixel), 2);
			arq_stream.read(reinterpret_cast<char*>(&arq_info.codigo_compressao), 4);
			arq_stream.read(reinterpret_cast<char*>(&arq_info.tamanho_imagem), 4);
			arq_stream.read(reinterpret_cast<char*>(&arq_info.res_hor), 4);
			arq_stream.read(reinterpret_cast<char*>(&arq_info.res_ver), 4);
			arq_stream.read(reinterpret_cast<char*>(&arq_info.cores_paleta), 4);
			arq_stream.read(reinterpret_cast<char*>(&arq_info.cores_importantes), 4);
    //verifica validade do cabeçalho
    //(creio que cada mensagem de erro explica o código)
    if(arq_info.formato[0]!='B' || arq_info.formato[1]!='M')
    {
      cout << "Arquivo " << nome_arquivo << " nao parece BMP" << endl;
    }
    else if(arq_info.offset_dados!=54 || arq_info.tamanho_info_header!=40)
    {
      cout << "Arquivo " << nome_arquivo << " tem formato nao suportado" << endl;
    }
    else if(arq_info.largura>2048 || arq_info.altura>2048)
    {
      cout << "Arquivo " << nome_arquivo << " representa imagem muito grande" << endl;
      cout << "(programa suporta ateh 2048 x 2048)" << endl;
    }
    else if(arq_info.largura<1 || arq_info.altura<1)
    {
      cout << "Arquivo " << nome_arquivo << " representa tamanho invalido" << endl;
      cout << "(as dimensoes sao nulas ou negativas, nao suportadas)" << endl;
    }
    else if(arq_info.tamanho_imagem<(static_cast<uint32_t>(arq_info.largura*arq_info.altura*3)))
    {
      cout << "Arquivo " << nome_arquivo << " corrompido" << endl;
      cout << "(Imagem nao possui pixels suficientes)" << endl;
    }
    else if(arq_info.bits_por_pixel!=24)
    {
      cout << "Arquivo " << nome_arquivo << " nao eh BMP 24 bits por pixel" << endl;
      cout << "(programa suporta apenas bitmaps de 24 bits por pixel)" << endl;
    }
    else if(arq_info.codigo_compressao!=0)
    {
      cout << "Arquivo " << nome_arquivo << " esta comprimido" << endl;
      cout << "(programa suporta apenas bitmaps sem compressao)" << endl;
    } else
    {
      //verifica se número de bytes na imagem corresponde com o esperado
      auto p0 = arq_stream.tellg();
      arq_stream.seekg(0, ios::end);
      auto p1 = arq_stream.tellg();
      if(static_cast<uint32_t>(p1-p0)!=(arq_info.tamanho_imagem))
      {
        cout << "Arquivo " << nome_arquivo << " parece corrompido" << endl;
        cout << "(Numero de bytes que informa nao eh o mesmo que possui)" << endl;
      }
      else
      {
        //se o arquivo parece OK, lê imagem; linha por linha
        //  define tamanho do resultado (cada bloco de 28*28 pixels é um caractere
					arq_imagem.resize((arq_info.altura/28)*(arq_info.largura/28));
        //tamanho que cada linha ocupa no arquivo
        size_t const linha_stride = arq_info.tamanho_imagem/arq_info.altura;
        size_t linha_pos = arq_info.offset_dados;
				int const letras_por_linha = arq_info.largura/28;
        //para cada linha
        for(int linha = (arq_info.altura-1); linha>=0; linha--)
        {
          //início da linha
          arq_stream.seekg(linha_pos);
          //para cada coluna
          for(int coluna =0; coluna<arq_info.largura; ++coluna)
          {
            uint8_t pixel[3];
            //tenta ler próximo pixel
            if(arq_stream.read(reinterpret_cast<char*>(pixel), 3))
            {
							//caractere a que pertence o pixel
								int const cx = coluna/28;
								int const cy = linha/28;
							//posição do pixel dentro do caractere
								int const px = coluna%28;
								int const py = linha%28;
              //em escala de cinza, todos componentes costumam são iguais a média
							// mesmo assim, a média dos pixels é calculada explicitamente aqui
								int32_t media = (pixel[0]+pixel[1]+pixel[2]+1)/3;
              arq_imagem[cy*letras_por_linha+cx][py][px] = media; // salva byte
            }
            else
            {
              //se falhar leitura, interrompe
              break;
            }
          }
          //se arquivo ok, avança linha
          if(arq_stream.good())
          {
            linha_pos += linha_stride;
          }
          else
          {
            //se falhou em meio a leitura, descarta toda imagem
            cout << "Ocorreu um erro ao ler " << nome_arquivo << endl;
            arq_imagem.clear();
            break;
          }
        }
      }
    }
  }
  //retorna vetor com imagem de caracteres em ordem
  return arq_imagem;
}

/// img_vazia() : função que identifica área preta da imagem
bool img_vazia(bm_ref_t const & img)
{
	int soma(0);
	for(auto & y : img )
	{
		if(soma > 2048 ) //2048 é uma margem arbitrária para considerar o bitmap nulo (caractere de espaço)
			{break;}
		for(auto x : y)
			{ soma += x; }
	}
	return (soma<=2048);
}

/**
 * identificar_letra : função que usa de rede para identificar caractere.
 * 
 * argumentos :
 * 	S : rede neural treinada para identificar letras
 *  img : array de 28*28 bytes, representando imagem de caractere em escala de cinza
 * 
 * a função verifica se a imagem é vazia, e retorna espaço em branco nesse caso
 * 
 */
char identificar_letra ( sistema_neural_t const & S, bm_ref_t img )
{
	//imagens completamente pretas são lidas como espaço em branco
		if(img_vazia(img))
			{ return ' ';}
	//transforma imagem em vetor de entrada
		auto x = gera_x_de_bitmap( img );
	//usa rede para resolver y
		auto y = S.resolve(x);
	//classifica y como classe mais provável
		int classe_i = mapa_y_classe(y);
	//retorna caractere associado a resultado. convertendo para minúsculas
		char r = mapa_EMNIST_treinado_letras[classe_i].second;
		if(r>='A' && r<='Z')
			{ r+= ('z'-'Z'); }
	return r;
}

/**
 * identificar_numero : função que usa de rede para identificar caractere numérico.
 * 
 * argumentos :
 * 	S : rede neural treinada para identificar dígitos numéricos
 *  img : array de 28*28 bytes, representando imagem de caractere em escala de cinza
 * 
 * a função verifica se a imagem é vazia, e retorna espaço em branco nesse caso
 * 
 */
char identificar_numero ( sistema_neural_t const & S, bm_ref_t img )
{
	//imagens completamente pretas são lidas como espaço em branco
		if(img_vazia(img))
			{ return ' ';}
	//transforma imagem em vetor de entrada
		auto x = gera_x_de_bitmap( img );
	//usa rede para resolver y
		auto y = S.resolve(x);
	//classifica y como classe mais provável
		int classe_i = mapa_y_classe(y);
	//retorna caractere associado a resultado
		char r = mapa_EMNIST_treinado_numeros[classe_i].second;
	return r;
}
