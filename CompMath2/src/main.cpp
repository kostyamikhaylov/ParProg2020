#include <iostream>
#include <iomanip>
#include <fstream>
#include <mpi.h>
#include <unistd.h>
#include <cmath>
#include <string.h>

#define OFFSET(y, x) (y) * xSize + (x)

void calc(double* frame, uint32_t ySize, uint32_t xSize, double delta, int rank, int size)
{
	MPI_Status status;
	MPI_Request request;
	MPI_Bcast (&xSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast (&ySize, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast (&delta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	uint32_t yFirst = ySize * rank / size;
	if (yFirst == 0)
		yFirst = 1;
	uint32_t yLast = ySize * (rank + 1) / size;
	if (yLast == ySize)
		yLast = ySize - 1;
	uint32_t yLen = yLast - yFirst;

	double diff = 0;
	if (yLen <= 0)
		diff = 2 * delta;
	double* tmpFrame = (double *) calloc (xSize * ySize, sizeof (*tmpFrame));
	if (rank)
		frame = (double *) calloc (xSize * ySize, sizeof (*frame));
	if (!tmpFrame || !frame)
	{
		fprintf (stderr, "calloc error\n");
		exit (-1);
	}

	MPI_Bcast (frame, xSize * ySize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	memcpy (tmpFrame, frame, xSize * ySize * sizeof (double));

	// Calculate first iteration
	for (uint32_t y = yFirst; y < yLast; y++)
	{
		for (uint32_t x = 1; x < xSize - 1; x++)
		{
			tmpFrame[OFFSET(y, x)] = (frame[OFFSET(y + 1, x)] + frame[OFFSET(y - 1, x)] + frame[OFFSET(y, x + 1)] + frame[OFFSET(y, x - 1)]) / 4.0;
			diff += std::abs (tmpFrame[OFFSET (y, x)] - frame[OFFSET(y, x)]);
		}
	}
	MPI_Barrier (MPI_COMM_WORLD);

	double* currFrame = tmpFrame;
	double* nextFrame = frame;
	uint32_t iteration = 1;
	// Calculate frames
	while (diff > delta)
	{
		diff = 0;
		double diff_reduce = 0;
		if (rank != 0 && yLen > 0)
		{
			MPI_Isend (currFrame + OFFSET (yFirst, 0), xSize, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &request);
			MPI_Irecv (currFrame + OFFSET (yFirst - 1, 0), xSize, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &request);
		}
		if (rank != size - 1 && yLen > 0)
		{
			MPI_Isend (currFrame + OFFSET (yLast - 1, 0), xSize, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &request);
			MPI_Irecv (currFrame + OFFSET (yLast, 0), xSize, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &request);
		}
		MPI_Barrier (MPI_COMM_WORLD);


		for (uint32_t y = yFirst; y < yLast; y++)
		{
			for (uint32_t x = 1; x < xSize - 1; x++)
			{
				nextFrame[OFFSET(y, x)] = (currFrame[OFFSET(y + 1, x)] + currFrame[OFFSET(y - 1, x)] + currFrame[OFFSET(y, x + 1)] + currFrame[OFFSET(y, x - 1)]) / 4.0;
				diff += std::abs (nextFrame[OFFSET(y, x)] - currFrame[OFFSET(y, x)]);
			}
		}

		MPI_Reduce (&diff, &diff_reduce, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		if (!rank)
			diff = diff_reduce;
		MPI_Bcast (&diff, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		std::swap (currFrame, nextFrame);
		iteration++;
	}

	// Copy result from tmp
	if (iteration % 2 == 1)
	{
		if (!rank)
		{
			memcpy (frame + OFFSET (yFirst, 0), tmpFrame + OFFSET (yFirst, 0), yLen * xSize * sizeof (double));
			for (int i = 1; i < size; i++)
			{
				uint32_t first = ySize * i / size;
				if (first == 0)
					first = 1;
				uint32_t last = ySize * (i + 1) / size;
				if (last == ySize)
					last = ySize - 1;
				uint32_t len = last - first;
				MPI_Recv (frame + OFFSET(first, 0), len * xSize, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
			}
		}
		if (rank)
			MPI_Send (tmpFrame + OFFSET (yFirst, 0), yLen * xSize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}
	else
	{
		if (!rank)
		{
			for (int i = 1; i < size; i++)
			{
				uint32_t first = ySize * i / size;
				if (first == 0)
					first = 1;
				uint32_t last = ySize * (i + 1) / size;
				if (last == ySize)
					last = ySize - 1;
				uint32_t len = last - first;
				MPI_Recv (frame + OFFSET(first, 0), len * xSize, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
			}
		}
		if (rank)
			MPI_Send (frame + OFFSET (yFirst, 0), yLen * xSize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}

	free (tmpFrame);
	if (rank)
		free (frame);
}

int main(int argc, char** argv)
{
  int rank = 0, size = 0, status = 0;
  double delta = 0;
  uint32_t ySize = 0, xSize = 0;
  double* frame = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0)
  {
    // Check arguments
    if (argc != 3)
    {
      std::cout << "[Error] Usage <inputfile> <output file>\n";
      status = 1;
      MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return 1;
    }

    // Prepare input file
    std::ifstream input(argv[1]);
    if (!input.is_open())
    {
      std::cout << "[Error] Can't open " << argv[1] << " for write\n";
      status = 1;
      MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return 1;
    }

    // Read arguments from input
    input >> ySize >> xSize >> delta;
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);

    frame = new double[ySize * xSize];

    for (uint32_t y = 0; y < ySize; y++)
    {
     for (uint32_t x = 0; x < xSize; x++)
      {
        input >> frame[y*xSize + x];
      }
    }
    input.close();
  } else {
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (status != 0)
    {
      return 1;
    }
  }

  calc(frame, ySize, xSize, delta, rank, size);

  if (rank == 0)
  {
    // Prepare output file
    std::ofstream output(argv[2]);
    if (!output.is_open())
    {
      std::cout << "[Error] Can't open " << argv[2] << " for read\n";
      delete frame;
      return 1;
    }
    for (uint32_t y = 0; y < ySize; y++)
    {
      for (uint32_t x = 0; x < xSize; x++)
      {
        output << " " << frame[y*xSize + x];
      }
      output << std::endl;
    }
    output.close();
    delete frame;
  }

  MPI_Finalize();
  return 0;
}
