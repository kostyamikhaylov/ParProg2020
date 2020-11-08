#include <iostream>
#include <iomanip>
#include <fstream>
#include <mpi.h>
#include <unistd.h>
#include <cmath>
#include <string.h>


#define OFFSET(x, y, z) ((z) * ySize * xSize + (y) * xSize + (x))

struct diag
{
	uint32_t x;
	uint32_t y;
	uint32_t z;
};

void calc(double* arr, uint32_t zSize, uint32_t ySize, uint32_t xSize, int rank, int size)
{
	double* reduce_arr = NULL, *backup_arr = NULL;
	MPI_Bcast (&xSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast (&ySize, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast (&zSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (rank)
		arr = (double *) calloc (xSize * ySize * zSize, sizeof (*arr));
	if (rank == 0)
	{
		backup_arr = (double *) calloc (xSize * ySize * zSize, sizeof (*backup_arr));
		reduce_arr = (double *) calloc (xSize * ySize * zSize, sizeof (*reduce_arr));
	}
	if (!arr || (!rank && (!backup_arr || !reduce_arr)))
	{
		fprintf (stderr, "Calloc error\n");
		exit (-1);
	}
	MPI_Bcast (arr, xSize * ySize * zSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	for (uint32_t z = 0; z < zSize; z++)
		for (uint32_t y = 0; y < ySize; y++)
			for (uint32_t x = 0; x < xSize; x++)
				if (x != xSize - 1 && y != ySize - 1 && z != 0)
					arr[OFFSET(x, y, z)] = 0;

	uint32_t xFirst = (xSize - 1) * (size - 1 - rank) / size;
	uint32_t xLast = (xSize - 1) * (size - 1 - rank + 1) / size;

	uint32_t yFirst = (ySize - 1) * (size - 1 - rank) / size;
	uint32_t yLast = (ySize - 1) * (size - 1 - rank + 1) / size;

	uint32_t zFirst = (zSize - 1) * rank / size + 1;
	uint32_t zLast = (zSize - 1) * (rank + 1) / size + 1;
	
	if (!rank)
		memcpy (backup_arr, arr, xSize * ySize * zSize * sizeof (double));

	uint32_t square = 0;
	for (uint32_t z = 1; z < zSize; z++)
		for (uint32_t y = 0; y < ySize - 1; y++)
			for (uint32_t x = 0; x < xSize - 1; x++)
			{
				if (!((x == xSize - 2) || (y == ySize - 2) || (z == 1)))
					continue;
				if ((z >= zFirst && z < zLast &&
							y >= yFirst && x >= xFirst) ||
					(y >= yFirst && y < yLast &&
							z < zLast && x >= xFirst) ||
					(x >= xFirst && x < xLast &&
							z < zLast && y >= yFirst))
					square++;
			}

	struct diag *diag = (struct diag *) calloc (square, sizeof (*diag));

	uint32_t ind = 0;

	for (uint32_t z = 1; z < zSize; z++)
		for (uint32_t y = 0; y < ySize - 1; y++)
			for (uint32_t x = 0; x < xSize - 1; x++)
			{
				if (!((x == xSize - 2) || (y == ySize - 2) || (z == 1)))
					continue;
				if ((z >= zFirst && z < zLast &&
							y >= yFirst && x >= xFirst) ||
					(y >= yFirst && y < yLast &&
							z < zLast && x >= xFirst) ||
					(x >= xFirst && x < xLast &&
							z < zLast && y >= yFirst))
				{
					diag[ind].x = x;
					diag[ind].y = y;
					diag[ind].z = z;
					ind++;
				}
			}
		
	for (ind = 0; ind < square; ind++)
	{
		uint32_t x = diag[ind].x;
		uint32_t y = diag[ind].y;
		uint32_t z = diag[ind].z;

		while (1)
		{
			arr[z * ySize * xSize + y * xSize + x] =
	sin(arr[(z - 1)* ySize * xSize + (y + 1) * xSize + x + 1]);
			if (x == 0 || y == 0 || z == zSize - 1)
				break;
			x--; y--; z++;
		}
	}

	free (diag);
	MPI_Reduce (arr, reduce_arr, xSize * ySize * zSize, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (rank == 0)
	{
		memcpy (arr, reduce_arr, xSize * ySize * zSize * sizeof (double));
		for (uint32_t z = 0; z < zSize; z++)
			for (uint32_t y = 0; y < ySize; y++)
				for (uint32_t x = 0; x < xSize; x++)
					if (x == xSize - 1 || y == ySize - 1 || z == 0)
					{
						arr[OFFSET(x, y, z)] = backup_arr[OFFSET(x, y, z)];
					}
		free (backup_arr);
		free (reduce_arr);
	}

	if (rank)
		free (arr);
}

int main(int argc, char** argv)
{
  int rank = 0, size = 0, buf = 0;
  uint32_t zSize = 0, ySize = 0, xSize = 0;
  double* arr = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0)
  {
    // Check arguments
    if (argc != 3)
    {
      std::cout << "[Error] Usage <inputfile> <output file>\n";
      buf = 1;
      MPI_Bcast(&buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return 1;
    }

    // Prepare input file
    std::ifstream input(argv[1]);
    if (!input.is_open())
    {
      std::cout << "[Error] Can't open " << argv[1] << " for write\n";
      buf = 1;
      MPI_Bcast(&buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return 1;
    }

    // Read arguments from input
    input >> zSize >> ySize >> xSize;
    MPI_Bcast(&buf, 1, MPI_INT, 0, MPI_COMM_WORLD);

    arr = new double[zSize * ySize * xSize];
    for (uint32_t z = 0; z < zSize; z++) {
      for (uint32_t y = 0; y < ySize; y++) {
        for (uint32_t x = 0; x < xSize; x++) {
          input >> arr[z*ySize*xSize + y*xSize + x];
        }
      }
    }
    input.close();
  } else {
    MPI_Bcast(&buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (buf != 0)
    {
      return 1;
    }
  }

  calc(arr, zSize, ySize, xSize, rank, size);

  if (rank == 0)
  {
    // Prepare output file
    std::ofstream output(argv[2]);
    if (!output.is_open())
    {
      std::cout << "[Error] Can't open " << argv[2] << " for read\n";
      delete arr;
      return 1;
    }

    for (uint32_t z = 0; z < zSize; z++) {
      for (uint32_t y = 0; y < ySize; y++) {
        for (uint32_t x = 0; x < xSize; x++) {
          output << " " << arr[z*ySize*xSize + y*xSize + x];
        }
        output << std::endl;
      }
      output << std::endl;
    }
    output.close();
    delete arr;
  }

  MPI_Finalize();
  return 0;
}
