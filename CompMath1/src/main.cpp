#include <iostream>
#include <iomanip>
#include <fstream>
#include <mpi.h>
#include <unistd.h>
#include <cmath>
#include <cstring>

double acceleration(double t)
{
	return sin (t);
}

void calc (double* trace, uint32_t traceSize, double t0, double dt, double y0, double y1, int rank, int size)
{
	MPI_Status status;
	MPI_Bcast (&traceSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast (&t0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast (&dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//every process calculates its working aria
	uint32_t first = traceSize * rank / size;
	uint32_t last = traceSize * (rank + 1) / size;
	uint32_t len = last - first;
	double v0 = 0.0;
	double start_y = 0.0, start_v = 0.0;
	double end_y = 0.0, end_v = 0.0;
	double prev_y = 0.0, prev_v = 0.0;
	if (!rank)
		start_y = y0;
	double* local_trace = (double *) calloc (len, sizeof (*local_trace));
	if (!local_trace)
	{
		fprintf (stderr, "calloc error\n");
		exit (-1);
	}
	
	//total time for each process
	double tau = dt * traceSize / size;
	//time shift (more convenient for acceleration() )
	t0 = t0 + tau * rank;

	// Sighting shot
	local_trace[0] = start_y;
	local_trace[1] = start_y + dt * start_v;
	for (uint32_t i = 2; i < len; i++)
	{
		local_trace[i] = dt * dt * acceleration (t0 + (i - 1) * dt)
					+ 2 * local_trace[i - 1] - local_trace[i - 2];
	}
	//save b_i and u_i (notation from seminar)
	end_y = local_trace[len - 1];
	end_v = (local_trace[len - 1] - local_trace[len - 2]) / dt;

	//if one proc only - sending no data
	if (size == 1)
	{
		v0 = (y1 - end_y) / (dt * traceSize);
		start_v = v0;
	}
	else	//otherwise consistently recalculate start and final values
	{
		if (rank) // == rank > 0
		{
			MPI_Recv (&prev_y, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &status);
			MPI_Recv (&prev_v, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &status);
			//set position and velocity as previous process' final values b_{i-1}, u_{i-1}
			start_y = prev_y;
			start_v = prev_v;
			//change own final values b_i, u_i
			end_y += prev_y + prev_v * tau;
			end_v += prev_v;
		}
		if (rank != size - 1)
		{
			//send final values b_i, u_i to the next process
			MPI_Send (&end_y, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
			MPI_Send (&end_v, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
		}

		if (rank == size - 1)
		{
			//the last process sends b* to the first
			MPI_Send (&end_y, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);	
		}
		if (!rank)
		{
			//the first get b* and calculates drift speed for all the processes
			MPI_Recv (&prev_y, 1, MPI_DOUBLE, size - 1, 0, MPI_COMM_WORLD, &status);
			v0 = (y1 - prev_y) / (dt * traceSize);
		}
		//the first broadcasts drift speed
		MPI_Bcast (&v0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		//every process takes the drift into account
		start_y += v0 * tau * rank;
		start_v += v0;
	}

	// The final shot

	local_trace[0] = start_y;
	local_trace[1] = start_y + dt * start_v;
	for (uint32_t i = 2; i < len; i++)
	{
		local_trace[i] = dt * dt * acceleration (t0 + (i - 1) * dt)
					+ 2 * local_trace[i - 1] - local_trace[i - 2];
	}

	//gather data from local_trace[] of all processes to trace[] of 1st process
	if (!rank)
	{
		memcpy (trace, local_trace, len * sizeof (double));
		for (int i = 1; i < size; i++)
		{
			uint32_t first = traceSize * i / size;
			uint32_t last = traceSize * (i + 1) / size;
			uint32_t len = last - first;
			MPI_Recv (trace + first, len, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
		}
	}
	if (rank)
		MPI_Send (local_trace, len, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);	

	free (local_trace);
}

int main(int argc, char** argv)
{
  int rank = 0, size = 0, status = 0;
  uint32_t traceSize = 0;
  double t0 = 0, t1 = 0, dt = 0, y0 = 0, y1 = 0;
  double* trace = 0;

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
    input >> t0 >> t1 >> dt >> y0 >> y1;
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    traceSize = (t1 - t0)/dt;
    trace = new double[traceSize];

    input.close();
  } else {
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (status != 0)
    {
      return 1;
    }
  }

  calc(trace, traceSize, t0, dt, y0, y1, rank, size);

  if (rank == 0)
  {
    // Prepare output file
    std::ofstream output(argv[2]);
    if (!output.is_open())
    {
      std::cout << "[Error] Can't open " << argv[2] << " for read\n";
      delete trace;
      return 1;
    }

    for (uint32_t i = 0; i < traceSize; i++)
    {
      output << " " << trace[i];
    }
    output << std::endl;
    output.close();
    delete trace;
  }

  MPI_Finalize();
  return 0;
}
