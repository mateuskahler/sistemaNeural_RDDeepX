
#include "emnist_leitor.hpp"
#include <fstream>
#include <sstream>

/**

	dos docs do MNIST:
	 
		LABEL FILE (idx1):
			[offset] [type]          [value]          [description] 
			0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
			0004     32 bit integer  60000            number of items 
			0008     unsigned byte   ??               label 
			0009     unsigned byte   ??               label 
			........ 
			xxxx     unsigned byte   ??               label
			The labels values are 0 to 9.

		IMAGE FILE (idx3):
			[offset] [type]          [value]          [description] 
			0000     32 bit integer  0x00000803(2051) magic number 
			0004     32 bit integer  60000            number of images 
			0008     32 bit integer  28               number of rows 
			0012     32 bit integer  28               number of columns 
			0016     unsigned byte   ??               pixel 
			0017     unsigned byte   ??               pixel 
			........ 
			xxxx     unsigned byte   ??               pixel
 */

std::pair<dig_refs_t, std::string> EMNIST_carrega_referencias(
	std::string arq_imgs, //caminho para arquivo de imagens
	std::string arq_etqs, //caminho para arquivo de etiquetas/labels
	bool transpoe_img
	)
{
	using std::uint8_t;
	using std::int8_t;
	using std::uint32_t;
	using std::int32_t;
	std::pair<dig_refs_t, std::string> Rs;
	//lambda para ler inteiro de 32 bits MSB
		auto ler_int32_idx = [](std::ifstream & arq)->int32_t
			{
				//lê cada byte e os agrupa em 32 bits em passos explícitos
				uint8_t  buf08[4];
				uint32_t buf32[4];
				for(int k=0; k<4; k++){
					arq.read(reinterpret_cast<char*>(&buf08[k]), 1);
					buf32[k] = buf08[k];}
				uint32_t comp  =  buf32[0] << 24 ;
								 comp |= (buf32[1] << 16);
								 comp |= (buf32[2] <<  8);
								 comp |=  buf32[3] ;
				return static_cast<int32_t>(comp);
			};
	//primeiro lê arquivo de etiquetas
	std::vector<int8_t> etiquetas;
	{
	std::ifstream arq(arq_etqs, std::ios::binary | std::ios::in);
		if(!arq)
		{
			std::stringstream ss;
			ss << "Erro ao abrir aquivo: " << arq_etqs;
			Rs.second = ss.str();
			return Rs;
		}
		//cabeçalho deve ser 2049 para formato idx1
			auto const cab = ler_int32_idx(arq);
			if((!arq) || (cab!=2049))
			{
				std::stringstream ss;
				ss << "Erro ao ler aquivo: " << arq_etqs;
				Rs.second = ss.str();
				return Rs;
			}
		//número de itens
			auto const arq_N = ler_int32_idx(arq);
			if((!arq) || (arq_N <0))
			{
				std::stringstream ss;
				ss << "Erro ao ler aquivo: " << arq_etqs;
				Rs.second = ss.str();
				return Rs;
			}
		//lê todas as etiquetas
			etiquetas.reserve(arq_N);
			for(int k=0; k<arq_N; k++)
			{
				int8_t buf;
				arq.read(reinterpret_cast<char*>(&buf), 1);
				if(!arq)
				{
					std::stringstream ss;
					ss << "Erro ao ler aquivo: " << arq_etqs;
					Rs.second = ss.str();
					return Rs;
				}
				etiquetas.emplace_back(buf);
			}
	} // fim do arquivo de etiquetas
			
	//segundo, lê arquivo de imagens
	{
	std::ifstream arq(arq_imgs, std::ios::binary | std::ios::in);
		if(!arq)
		{
			std::stringstream ss;
			ss << "Erro ao abrir aquivo: " << arq_imgs;
			Rs.second = ss.str();
			return Rs;
		}
		//cabeçalho deve ser 2051 para formato idx3
			auto const cab = ler_int32_idx(arq);
			if((!arq) || (cab!=2051))
			{
				std::stringstream ss;
				ss << "Erro ao ler aquivo: " << arq_imgs;
				Rs.second = ss.str();
				return Rs;
			}
		//número de itens
			auto const arq_N = ler_int32_idx(arq);
			if((!arq) || (arq_N <0))
			{
				std::stringstream ss;
				ss << "Erro ao ler aquivo: " << arq_imgs;
				Rs.second = ss.str();
				return Rs;
			}
			if((arq_N != int(etiquetas.size())))
			{
				std::stringstream ss;
				ss << "Numero de etiquetas != numero de imagens ( " << etiquetas.size() << " != " << arq_N << " )";
				Rs.second = ss.str();
				return Rs;
			}
		//dimensão das imagens
			auto const imgY = ler_int32_idx(arq);
			auto const imgX = ler_int32_idx(arq);
			if((!arq) || (arq_N <0))
			{
				std::stringstream ss;
				ss << "Erro ao ler aquivo: " << arq_imgs;
				Rs.second = ss.str();
				return Rs;
			}
			if((imgY != 28) || (imgX != 28))
			{
				std::stringstream ss;
				ss << "Tamnho de imagens incompativel " << imgX << " x " << imgY;
				Rs.second = ss.str();
				return Rs;
			}
		//lê todas as imagens e junta com etiqueta
			Rs.first.reserve(arq_N);
			for(int k=0; k<arq_N; k++)
			{
				dig_ref_t r;
				r.etiqueta = etiquetas[k];
				//lê uma linha de cada imagem por vez
				int8_t buf[28];
				for(int y = 0; y<28; y++)
				{
					arq.read(reinterpret_cast<char*>(&buf[0]), 28);
					if(!arq)
					{
						std::stringstream ss;
						ss << "Erro ao ler aquivo: " << arq_imgs;
						Rs.second = ss.str();
						return Rs;
					}
					//copia linha/coluna para resultado
					for(int x = 0; x<28; x++)
					{
						if(transpoe_img)
						{
							r.img[x][y] = buf[x];
						}
						else
						{
							r.img[y][x] = buf[x];
						}
					}
				}
				Rs.first.emplace_back(r);
			}
	} // fim do arquivo de imagens
	
	//gera string de status:
		std::stringstream ss;
		ss << (int)(Rs.first.size()) << " imagens carregadas";
		Rs.second = ss.str();
		
	return Rs;
}
