#include <iostream>
#include <iomanip>
#include <fstream>
#include <mpi.h>
#include <unistd.h>
#include <cmath>
#include <string.h>

void calc(double* arr, uint32_t ySize, uint32_t xSize, int rank, int size)
{
	double* reduce = NULL, *backup_arr = NULL;
	MPI_Bcast (&xSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast (&ySize, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (rank)
		arr = (double *) calloc (xSize * ySize, sizeof (*arr));
	if (rank == 0)
	{
		backup_arr = (double *) calloc (xSize * ySize, sizeof (*backup_arr));
		reduce = (double *) calloc (xSize * ySize, sizeof (*reduce));
	}
	if (!arr || (!rank && !backup_arr) || (!rank && !reduce))
	{
		fprintf (stderr, "Calloc error\n");
		exit (-1);
	}
	MPI_Bcast (arr, xSize * 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	uint32_t xFirst = xSize * rank / size;
	uint32_t xLast = xSize * (rank + 1) / size;

	if (!rank)
	{
		memcpy (backup_arr, arr, xSize * ySize * sizeof (double));
		memset (arr + xSize * 4, '0', xSize * (ySize - 4) * sizeof (double));
	}

	for (uint32_t y = 4; y < ySize; y++)
	{
		for (uint32_t x = xFirst; x < xLast; x++)
		{
			arr[y * xSize + x] = sin (arr[(y - 4) * xSize + x]);
		}
	}

	MPI_Reduce (arr, reduce, xSize * ySize, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (rank == 0)
	{
		for (uint32_t y = 0; y < 4; y++)
		{
			memcpy (reduce + y * xSize, backup_arr + y * xSize, xSize * sizeof (double));
		}
		free (backup_arr);
		memcpy (arr, reduce, xSize * ySize * sizeof (double));
		free (reduce);
	}

	if (rank)
		free (arr);
}

int main(int argc, char** argv)
{
  int rank = 0, size = 0, buf = 0;
  uint32_t ySize = 0, xSize = 0;
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
    input >> ySize >> xSize;
    MPI_Bcast(&buf, 1, MPI_INT, 0, MPI_COMM_WORLD);

    arr = new double[ySize * xSize];

    for (uint32_t y = 0; y < ySize; y++)
    {
     for (uint32_t x = 0; x < xSize; x++)
      {
        input >> arr[y*xSize + x];
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

  calc(arr, ySize, xSize, rank, size);

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
    for (uint32_t y = 0; y < ySize; y++)
    {
      for (uint32_t x = 0; x < xSize; x++)
      {
        output << " " << arr[y*xSize + x];
      }
      output << std::endl;
    }
    output.close();
    delete arr;
  }

  MPI_Finalize();
  return 0;
}
