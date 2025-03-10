#include <vector>
#include <Windows.h>
#include <chrono>
#include <iostream>

using namespace std;

using vd = vector<double>;
using vvd = vector<vector<double>>;

#define NUM_OF_THREADS 4

void fill_matrix(vvd& mx, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			mx[i][j] = (rand() % 100) * 1e-2;
		}
	}
}

void simple_mult(const vvd& a, const vvd& b, vvd& c, int N) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}
}

void omp_mult(const vvd& A, const vvd& B, vvd& C, int n) {
	#pragma omp parallel for num_threads(NUM_OF_THREADS)
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			double sum = 0;
			#pragma omp simd
			for (int k = 0; k < n; k++) {
				sum += A[i][k] * B[k][j];
			}
			C[i][j] = sum;
		}
	}
}

struct ThreadData {
	const vvd* A;
	const vvd* B;
	vvd* C;
	int N;
	int start_row;
	int end_row;
	int num;
};

DWORD WINAPI multiply_matrices_thread(LPVOID param) {
	ThreadData* data = (ThreadData*)param;
	for (int i = data->start_row; i < data->end_row; ++i) {
		for (int j = 0; j < data->N; ++j) {
			(*data->C)[i][j] = 0;
			for (int k = 0; k < data->N; ++k) {
				(*data->C)[i][j] += (*data->A)[i][k] * (*data->B)[k][j];
			}
		}
	}
	return 0;
}

void winapi_mult(const vvd& A, const vvd& B, vvd& C, int N) {
	const int num_threads = NUM_OF_THREADS;
	HANDLE threads[num_threads];
	ThreadData thread_data[num_threads];

	int rows_per_thread = N / num_threads;
	for (int i = 0; i < num_threads; ++i) {
		thread_data[i] = { &A, &B, &C, N, i * rows_per_thread, (i + 1) * rows_per_thread };
		threads[i] = CreateThread(NULL, 0, multiply_matrices_thread, &thread_data[i], 0, NULL);
	}

	WaitForMultipleObjects(num_threads, threads, TRUE, INFINITE);
	for (int i = 0; i < num_threads; ++i) {
		CloseHandle(threads[i]);
	}
}


int main() {
	vector<int> ns = { 1, 10, 50, 100, 250, 500, 1000, 1500, 2000, 2500, 3000 };
	for (auto n : ns) {
		cout << "N: " << n << endl;
		vvd a(n, vd(n, 0));
		vvd b(n, vd(n, 0));
		vvd c(n, vd(n, 0));
		fill_matrix(a, n);
		fill_matrix(b, n);
		auto start = chrono::high_resolution_clock::now();
		simple_mult(a, b, c, n);
		auto end = chrono::high_resolution_clock::now();
		chrono::duration<double> dur = end - start;
		cout << "Naive time : " << dur.count() << " sec" << endl;

		c = vvd(n, vd(n, 0));
		start = chrono::high_resolution_clock::now();
		omp_mult(a, b, c, n);
		end = chrono::high_resolution_clock::now();
		chrono::duration<double> durOMP = end - start;
		cout << "OpenMP time : " << durOMP.count() << " sec" << endl;
		cout << "OpenMP speedup : " << dur.count() / durOMP.count() << endl;
		cout << "OpenMP efficiency: " << dur.count() / durOMP.count() / (NUM_OF_THREADS / 2) << endl;

		c = vvd(n, vd(n, 0));
		start = chrono::high_resolution_clock::now();
		winapi_mult(a, b, c, n);
		end = chrono::high_resolution_clock::now();
		chrono::duration<double> durWinAPI = end - start;
		cout << "WinAPI time: " << durWinAPI.count() << " sec" << endl;
		cout << "WinAPI speedup : " << dur.count() / durWinAPI.count() << endl;
		cout << "WinAPI efficiency: " << dur.count() / durWinAPI.count() / (NUM_OF_THREADS / 2) << endl;
	}
}