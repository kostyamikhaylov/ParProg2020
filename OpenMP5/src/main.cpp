#include <iostream>
#include <iomanip>
#include <fstream>
#include <omp.h>

unsigned int CountNeighbours (uint32_t xSize, uint32_t ySize, uint8_t *inputFrame, uint32_t x, uint32_t y);

void calc (uint32_t xSize, uint32_t ySize, uint32_t iterations, uint32_t num_threads, uint8_t* inputFrame, uint8_t* outputFrame)
{
	unsigned int len = xSize * ySize;
	uint32_t num = 0, x = 0, y = 0;
	uint8_t tmp[len];
	
	#pragma omp parallel num_threads (num_threads) private (x, y, num)
	{
		for (uint32_t t = 0; t < iterations; t++)
		{
			#pragma omp for
			for (x = 0; x < xSize; x++)
				for (y = 0; y < xSize; y++)
				{
					num = CountNeighbours (xSize, ySize, inputFrame, x, y);
					if (num == 3)
						tmp[x * xSize + y] = 1;
					else if (num == 2 && inputFrame[x * xSize + y] == 1)
						tmp[x * xSize + y] = 1;
					else
						tmp[x * xSize + y] = 0;
				}

			#pragma omp for
			for (x = 0; x < xSize; x++)
				for (y = 0; y < xSize; y++)
					inputFrame[x * xSize + y] = tmp[x * xSize + y];
		}

		#pragma omp barrier
		
		#pragma omp for
		for (x = 0; x < xSize; x++)
			for (y = 0; y < xSize; y++)
			{
				if (iterations)
					outputFrame[x * xSize + y] = tmp[x * xSize + y];
				else
					outputFrame[x * xSize + y] = inputFrame[x * xSize + y];
			}
		#pragma omp barrier
	}

	return ;	
}

unsigned int CountNeighbours (uint32_t xSize, uint32_t ySize, uint8_t *inputFrame, uint32_t x, uint32_t y)
{
	uint32_t xm = 0, xp = 0, ym = 0, yp = 0;
	
	if (x == 0)
	{
		xm = xSize - 1;
		xp = x + 1;
	}
	else if (x == xSize - 1)
	{
		xm = xSize - 2;
		xp = 0;
	}
	else
	{
		xm = x - 1;
		xp = x + 1;
	}
	
	if (y == 0)
	{
		ym = ySize - 1;
		yp = y + 1;
	}
	else if (y == ySize - 1)
	{
		ym = ySize - 2;
		yp = 0;
	}
	else
	{
		ym = y - 1;
		yp = y + 1;
	}
	
	return	(inputFrame[xm * xSize + ym] == 1)
		+ 	(inputFrame[xm * xSize + y] == 1)
		+ 	(inputFrame[xm * xSize + yp] == 1)
		+ 	(inputFrame[x * xSize + ym] == 1)
		+ 	(inputFrame[x * xSize + yp] == 1)
		+ 	(inputFrame[xp * xSize + ym] == 1)
		+ 	(inputFrame[xp * xSize + y] == 1)
		+ 	(inputFrame[xp * xSize + yp] == 1);
}

int main(int argc, char** argv)
{
  // Check arguments
  if (argc != 3)
  {
    std::cout << "[Error] Usage <inputfile> <output file>\n";
    return 1;
  }

  // Prepare input file
  std::ifstream input(argv[1]);
  if (!input.is_open())
  {
    std::cout << "[Error] Can't open " << argv[1] << " for write\n";
    return 1;
  }

  // Prepare output file
  std::ofstream output(argv[2]);
  if (!output.is_open())
  {
    std::cout << "[Error] Can't open " << argv[2] << " for read\n";
    input.close();
    return 1;
  }

  // Read arguments from input
  uint32_t xSize = 0, ySize = 0, iterations = 0, num_threads = 0;
  input >> xSize >> ySize >> iterations >> num_threads;
  uint8_t* inputFrame = new uint8_t[xSize*ySize];
  uint8_t* outputFrame = new uint8_t[xSize*ySize];
  for (uint32_t y = 0; y < ySize; y++)
  {
    for (uint32_t x = 0; x < xSize; x++)
    {
      input >> inputFrame[y*xSize + x];
      inputFrame[y*xSize + x] -= '0';
    }
  }

  // Calculation
  calc(xSize, ySize, iterations, num_threads, inputFrame, outputFrame);

  // Write result
  for (uint32_t y = 0; y < ySize; y++)
  {
    for (uint32_t x = 0; x < xSize; x++)
    {
      outputFrame[y*xSize + x] += '0';
      output << " " << outputFrame[y*xSize + x];
    }
    output << "\n";
  }

  // Prepare to exit
  delete outputFrame;
  delete inputFrame;
  output.close();
  input.close();
  return 0;
}
