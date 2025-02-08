#include "Parallel.h"

#include <omp.h>
#include <iostream>

namespace ML
{

  void spreadProcess()
  {

  int tid;
  int nThreads;

#pragma omp parallel private(tid)
    {

      // This statement will run on each thread.
      // If there are 4 threads, this will execute 4 times in total
      tid = omp_get_thread_num();
      printf("Running on thread %d\n", tid);

      if (tid == 0)
      {
        nThreads = omp_get_num_threads();
        printf("Total number of threads: %d\n", nThreads);
      }

      try
      {
        auto env_val = std::getenv("API_KEY");
        if (env_val != nullptr)
          std::cout << "ENV API_KEY VAL = " << env_val << "\n";
      }
      catch (std::exception &e)
      {
        std::cout << "Except: " << e.what() << "\n";
        exit(1);
      }
    }
  }
}