# include <vector>
# include <math.h>
using namespace std;
# include <iostream>

vector<vector<double>> make_zero_martix(int m, int n) {
	//创建0矩阵
	vector<vector<double>> array;
	vector<double> temparay;
	for (int i = 0; i < m; ++i)// m*n 维数组
	{
		for (int j = 0; j < n; ++j)
			temparay.push_back(i * j);
		array.push_back(temparay);
		temparay.erase(temparay.begin(), temparay.end());
	}
	return array;
}
//按第一行展开计算|A|
double getA(vector<vector<double>> arcs, int n)
{
	if (n == 1)
	{
		return arcs[0][0];
	}
	double ans = 0;
	vector<vector<double>> temp = make_zero_martix(arcs.size(), arcs.size());
	int i, j, k;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n - 1; j++)
		{
			for (k = 0; k < n - 1; k++)
			{
				temp[j][k] = arcs[j + 1][(k >= i) ? k + 1 : k];
			}
		}
		double t = getA(temp, n - 1);
		if (i % 2 == 0)
		{
			ans += arcs[0][i] * t;
		}
		else
		{
			ans -= arcs[0][i] * t;
		}
	}
	return ans;
}
//计算每一行每一列的每个元素所对应的余子式，组成A*
void  getAStart(vector<vector<double>> arcs, int n, vector<vector<double>>& ans)
{
	if (n == 1)
	{
		ans[0][0] = 1;
		return;
	}
	int i, j, k, t;
	vector<vector<double>> temp = make_zero_martix(n, n);
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			for (k = 0; k < n - 1; k++)
			{
				for (t = 0; t < n - 1; t++)
				{
					//cout << arcs[k >= i ? k + 1 : k][t >= j ? t + 1 : t] << endl;
					temp[k][t] = arcs[k >= i ? k + 1 : k][t >= j ? t + 1 : t];
				}
			}
			ans[j][i] = getA(temp, n - 1);  //此处顺便进行了转置
			if ((i + j) % 2 == 1)
			{
				ans[j][i] = -ans[j][i];
			}
		}
	}
}
//得到给定矩阵src的逆矩阵保存到des中。
bool GetMatrixInverse(vector<vector<double>> src, int n, vector<vector<double>>& des)
{
	double flag = getA(src, n);
	vector<vector<double>> t = make_zero_martix(n, n);
	if (0 == flag)
	{
		cout << "原矩阵行列式为0，无法求逆。请重新运行" << endl;
		return false;//如果算出矩阵的行列式为0，则不往下进行
	}
	else
	{
		getAStart(src, n, t);
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
			{
				des[i][j] = t[i][j] / flag;
			}
		}
	}
	return true;
}
vector<vector<double>> m_inverse(vector<vector<double>> matrix_before) {
	//矩阵求逆
	bool flag;
	vector<vector<double>> matrix_after = make_zero_martix(matrix_before.size(), matrix_before.size());
	flag = GetMatrixInverse(matrix_before, matrix_before.size(), matrix_after);
	return matrix_after;
}
// 矩阵运算基本
const double epsilon=1e-14;  //小于该数判断为0
//创建 h行l列的矩阵，并将初始各值设定为0
vector<vector<double>> creatmatrix(int h, int l)
{
	vector<vector<double>> v;
	for (int i = 0; i < h; i++)
	{
		vector<double>v1(l, 0);
		v.push_back(v1);
	}
	return v;
}
//矩阵A+矩阵B=矩阵C，并返回
vector<vector<double>> m_plus(const vector<vector<double>>& A, const vector<vector<double>>& B)
{
	int h = A.size();
	int l = A[0].size();
	vector<vector<double>> C;
	C = creatmatrix(h, l);

	for (int i = 0;i < h;i++)
	{
		for (int j = 0; j < l; j++)
		{
			C[i][j] = A[i][j] + B[i][j];
			if (abs(C[i][j]) < epsilon)
			{
				C[i][j] = 0.0;
			}
		}
	}
	return C;
}
//矩阵A-矩阵B=矩阵C，并返回
vector<vector<double>> m_minus(const vector<vector<double>>& A, const vector<vector<double>>& B)
{
	int h = A.size();
	int l = A[0].size();
	vector<vector<double>> C;
	C = creatmatrix(h, l);

	for (int i = 0;i < h;i++)
	{
		for (int j = 0; j < l; j++)
		{
			C[i][j] = A[i][j] - B[i][j];
			if (abs(C[i][j]) < epsilon)
			{
				C[i][j] = 0.0;
			}
		}
	}
	return C;
}
//矩阵A*矩阵B=矩阵C，并返回
vector<vector<double>> multiply(const vector<vector<double>>& A, const vector<vector<double>>& B)
{
	int A_h = A.size();
	int A_l = A[0].size();
	int B_h = B.size();
	int B_l = B[0].size();
	if (A_l != B_h)
	{
		cout << "两矩阵维数无法相乘" << endl;
		exit(0);
	}
	vector<vector<double>> C = creatmatrix(A_h, B_l);
	for (int i = 0; i < A_h; i++)
	{
		for (int j = 0; j < B_l; j++)
		{
			C[i][j] = 0;
			for (int k = 0; k < A_l; k++)
			{
				C[i][j] += A[i][k] * B[k][j];
			}
			if (abs(C[i][j]) < epsilon)
			{
				C[i][j] = 0.0;
			}
			//cout<<C[i][j]<<"\t";
		}
		//cout<<endl;
	}
	return C;
}
//矩阵A*num=矩阵B，并返回
vector<vector<double>> multiply_num(const vector<vector<double>>& A, double num)
{
	int A_h = A.size();
	int A_l = A[0].size();
	vector<vector<double>> B = creatmatrix(A_h, A_l);
	for (int i = 0; i < A_h; i++)
	{
		for (int j = 0; j < A_l; j++)
		{
			B[i][j] = num * A[i][j];
		}
	}
	return B;
}
//矩阵A与矩阵B上下叠加获得新的矩阵C,并返回
vector<vector<double>> matrix_overlaying_below(const vector<vector<double>>& A, const vector<vector<double>>& B)
{
	//判断矩阵的列是否相等
	int A_h = A.size();
	int A_l = A[0].size();
	int B_h = B.size();
	int B_l = B[0].size();
	if (A_l != B_l)
	{
		cout << "叠加的矩阵列数不相等" << endl;
		exit(0);
	}
	//创建
	vector<vector<double>> C = creatmatrix(A_h + B_h, A_l);
	//将A传入
	for (int i = 0; i < A_h; i++)
	{
		for (int j = 0; j < A_l; j++)
		{
			C[i][j] = A[i][j];
		}
	}
	//将B传入
	for (int i = 0; i < B_h; i++)
	{
		for (int j = 0; j < B_l; j++)
		{
			C[i + A_h][j] = B[i][j];
		}
	}
	return C;
}
//矩阵A与矩阵B左右叠加，获得新的矩阵C，并返回
vector<vector<double>> matrix_overlaying_beside(const vector<vector<double>>& A, const vector<vector<double>>& B)
{
	//判断矩阵的列是否相等
	int A_h = A.size();
	int A_l = A[0].size();
	int B_h = B.size();
	int B_l = B[0].size();
	if (A_h != B_h)
	{
		cout << "叠加的矩阵行数不相等" << endl;
		exit(0);
	}
	//创建
	vector<vector<double>> C = creatmatrix(A_h, A_l + B_l);
	//将A传入
	for (int i = 0; i < A_h; i++)
	{
		for (int j = 0; j < A_l; j++)
		{
			C[i][j] = A[i][j];
		}
	}
	//将B传入
	for (int i = 0; i < B_h; i++)
	{
		for (int j = 0; j < B_l; j++)
		{
			C[i][j + A_l] = B[i][j];
		}
	}
	return C;
}
//输入矩阵A，输出矩阵A的转置矩阵AT
vector<vector<double>> trans(const vector<vector<double>>& A)
{
	vector<vector<double>> AT = creatmatrix(A[0].size(), A.size());
	int h = AT.size();
	int l = AT[0].size();
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < l; j++)
		{
			AT[i][j] = A[j][i];
		}
	}
	return AT;
}
//输入矩阵A,输出矩阵A的逆矩阵inv_A
vector<vector<double>> inverse(const vector<vector<double>>& A)
{
	if (A.size() != A[0].size())
	{
		cout << "输入矩阵维数不合法" << endl;
		exit(0);
	}
	int n = A.size();
	vector<vector<double>> inv_A = creatmatrix(n, n);
	vector<vector<double>> L = creatmatrix(n, n);
	vector<vector<double>> U = creatmatrix(n, n);
	vector<vector<double>> inv_L = creatmatrix(n, n);
	vector<vector<double>> inv_U = creatmatrix(n, n);
	//LU分解
		//L矩阵对角元素为1
	for (int i = 0; i < n; i++)
	{
		L[i][i] = 1;
	}
	//U矩阵第一行
	for (int i = 0; i < n; i++)
	{
		U[0][i] = A[0][i];
	}
	//L矩阵第一列
	for (int i = 1; i < n; i++)
	{
		L[i][0] = 1.0 * A[i][0] / A[0][0];
	}

	//计算LU上下三角
	for (int i = 1; i < n; i++)
	{
		//计算U（i行j列）
		for (int j = i; j < n; j++)
		{
			double tem = 0;
			for (int k = 0; k < i; k++)
			{
				tem += L[i][k] * U[k][j];
			}
			U[i][j] = A[i][j] - tem;
			if (abs(U[i][j]) < epsilon)
			{
				U[i][j] = 0.0;
			}
		}
		//计算L（j行i列）
		for (int j = i; j < n; j++)
		{
			double tem = 0;
			for (int k = 0; k < i; k++)
			{
				tem += L[j][k] * U[k][i];
			}
			L[j][i] = 1.0 * (A[j][i] - tem) / U[i][i];
			if (abs(L[i][j]) < epsilon)
			{
				L[i][j] = 0.0;
			}
		}

	}
	//L U剩余位置设为0
	for (int i = 0;i < n;i++)
	{
		for (int j = 0;j < n;j++)
		{
			if (i > j)
			{
				U[i][j] = 0.0;
			}
			else if (i < j)
			{
				L[i][j] = 0.0;
			}
		}
	}
	//LU求逆
	//求矩阵U的逆 
	for (int i = 0;i < n;i++)
	{
		inv_U[i][i] = 1 / U[i][i];// U对角元素的值，直接取倒数
		for (int k = i - 1;k >= 0;k--)
		{
			double s = 0;
			for (int j = k + 1;j <= i;j++)
			{
				s = s + U[k][j] * inv_U[j][i];
			}
			inv_U[k][i] = -s / U[k][k];//迭代计算，按列倒序依次得到每一个值，
			if (abs(inv_U[k][i]) < epsilon)
			{
				inv_U[k][i] = 0.0;
			}
		}
	}
	//求矩阵L的逆
	for (int i = 0;i < n;i++)
	{
		inv_L[i][i] = 1; //L对角元素的值，直接取倒数，这里为1
		for (int k = i + 1;k < n;k++)
		{
			for (int j = i;j <= k - 1;j++)
			{
				inv_L[k][i] = inv_L[k][i] - L[k][j] * inv_L[j][i];
				if (abs(inv_L[k][i]) < epsilon)
				{
					inv_L[k][i] = 0.0;
				}
			}
		}
	}
	inv_A = multiply(inv_U, inv_L);
	return inv_A;
}
void show_matrix(const vector<vector<double>>& A)
{
	int h = A.size();
	int l = A[0].size();
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < l; j++)
		{
			cout << A[i][j] << "\t";
		}
		cout << endl;
	}
}

//////////////////////////////以上为实现矩阵基本运算的开源代码//////////////////////////////

// 计算旋转矩阵 
vector<vector<double>> cal_R(double p, double w, double k) 
{
	//vector<vector<double>> a = creatmatrix(1, 3);
	//vector<vector<double>> b = creatmatrix(1, 3);
	//vector<vector<double>> c = creatmatrix(1, 3);

	//double a2 = a[0][1] = -cos(p) * sin(k) - sin(p) * sin(w) * cos(k);
	//double a3 = a[0][2] = -sin(p) * cos(w);
	//double b3 = b[0][2] = -sin(w);

	//double a1 = a[0][0] = sqrt(1 - a2 * a2 - a3 * a3);
	//double c3 = c[0][2] = sqrt(1 - a3 * a3 - b3 * b3);

	////double b1 = b[0][0] = (-a1 * a3 * b3 - a2 * c3) / (1 - a3 * a3);
	//double b1 = b[0][0] = -0.998581;
	//double b2 = b[0][1] = sqrt(1 - b1 * b1 - b3 * b3);
	//c[0][0] = a2 * b3 - a3 * b2;
	//c[0][1] = a3 * b1 - a1 * b3;

	//vector<vector<double>> R = matrix_overlaying_below(a,b);
	//R = matrix_overlaying_below(R,c);
	//show_matrix(R);

	vector<vector<double>> R = creatmatrix(3, 3);
	R[0][0] = cos(p) * cos(k) - sin(p) * sin(w) * sin(k);
	R[0][1] = -cos(p) * sin(k) - sin(p) * sin(w) * cos(k);
	R[0][2] = -sin(p) * cos(w);
	R[1][0] = cos(w) * sin(k);
	R[1][1] = cos(w) * cos(k);
	R[1][2] = -sin(w);
	R[2][0] = sin(p) * cos(k) + cos(p) * sin(w) * sin(k);
	R[2][1] = -sin(p) * sin(k) + cos(p) * sin(w) * cos(k);
	R[2][2] = cos(p) * cos(w);

	return R;
}


// 正向计算像点坐标
	// 计算新x
vector<vector<double>> to_x(vector<vector<double>> X, vector<vector<double>> Y, vector<vector<double>> Z, double p, double w, double k, double f,double Xs0, double Ys0, double Zs0) 
{
	vector<vector<double>> R = cal_R(p, w, k);
	double a1 = R[0][0];
	double a2 = R[0][1];
	double a3 = R[0][2];

	double b1 = R[1][0];
	double b2 = R[1][1];
	double b3 = R[1][2];

	double c1 = R[2][0];
	double c2 = R[2][1];
	double c3 = R[2][2];
	
	//vector<vector<double>> Xs = creatmatrix(1, 4);
	//vector<vector<double>> Ys = creatmatrix(1, 4);
	//vector<vector<double>> Zs = creatmatrix(1, 4);
	//Xs[0] = { Xs0, Xs0, Xs0, Xs0 };
	//Ys[0] = { Ys0, Ys0, Ys0, Ys0 };
	//Zs[0] = { Zs0, Zs0, Zs0, Zs0 };

	//Xs = trans(Xs);
	//Ys = trans(Ys);
	//Zs = trans(Zs);

	//vector<vector<double>> up = m_plus(m_plus(multiply_num(m_minus(X, Xs), a1),multiply_num(m_minus(Y, Ys), b1)),multiply_num(m_minus(Z, Zs), c1));
	//vector<vector<double>> down = m_plus(m_plus(multiply_num(m_minus(X, Xs), a3),multiply_num(m_minus(Y, Ys), b3)),multiply_num(m_minus(Z, Zs), c3));

	//vector<vector<double>> x = up;
	//// down的各个单位求倒并且乘以up
	//for (int i = 0; i < down.size(); i++) 
	//{

	//	x[i][0] = up[i][0]/down[i][0];
	//}
	//
	//x = multiply_num(x, -f);
	//show_matrix(x);
	vector<vector<double>> x = creatmatrix(X.size(),1);
	for (int i = 0; i <	X.size(); i++)
	{
		x[i][0] = -f * (a1 * (X[i][0] - Xs0) + b1 * (Y[i][0] - Ys0) + c1 * (Z[i][0] - Zs0)) / (a3 * (X[i][0] - Xs0) + b3 * (Y[i][0] - Ys0) + c3 * (Z[i][0] - Zs0));
	}
	//show_matrix(x);
	return x;
}

vector<vector<double>> to_y(vector<vector<double>> X, vector<vector<double>> Y, vector<vector<double>> Z, double p, double w, double k, double f, double Xs0, double Ys0, double Zs0)
{
	vector<vector<double>> R = cal_R(p, w, k);
	double a1 = R[0][0];
	double a2 = R[0][1];
	double a3 = R[0][2];

	double b1 = R[1][0];
	double b2 = R[1][1];
	double b3 = R[1][2];

	double c1 = R[2][0];
	double c2 = R[2][1];
	double c3 = R[2][2];

	//vector<vector<double>> Xs = creatmatrix(1, 4);
	//vector<vector<double>> Ys = creatmatrix(1, 4);
	//vector<vector<double>> Zs = creatmatrix(1, 4);
	//Xs[0] = { Xs0, Xs0, Xs0, Xs0 };
	//Ys[0] = { Ys0, Ys0, Ys0, Ys0 };
	//Zs[0] = { Zs0, Zs0, Zs0, Zs0 };

	//Xs = trans(Xs);
	//Ys = trans(Ys);
	//Zs = trans(Zs);

	//vector<vector<double>> up = m_plus(m_plus(multiply_num(m_minus(X, Xs), a2), multiply_num(m_minus(Y, Ys), b2)), multiply_num(m_minus(Z, Zs), c2));
	//vector<vector<double>> down = m_plus(m_plus(multiply_num(m_minus(X, Xs), a3), multiply_num(m_minus(Y, Ys), b3)), multiply_num(m_minus(Z, Zs), c3));
	//
	//vector<vector<double>> y = up;
	//// down的各个单位求倒并且乘以up
	//for (int i = 0; i < down.size(); i++)
	//{

	//	y[i][0] = up[i][0] / down[i][0];
	//}

	//y = multiply_num(y, -f);
	//show_matrix(y);

	vector<vector<double>> y = creatmatrix(X.size(), 1);
	for (int i = 0; i < X.size(); i++)
	{
		y[i][0] = -f * (a2 * (X[i][0] - Xs0) + b2 * (Y[i][0] - Ys0) + c2 * (Z[i][0] - Zs0)) / (a3 * (X[i][0] - Xs0) + b3 * (Y[i][0] - Ys0) + c3 * (Z[i][0] - Zs0));
	}
	//show_matrix(y);
	return y;
}

// 构建误差方程和法方程
	// 计算A
vector<vector<double>> to_A(double X, double Y, double Z,double p, double w, double k, double f, double Xs, double Ys, double Zs,double x,double y)
{
	vector<vector<double>> R = cal_R(p, w, k);
	double a1 = R[0][0];
	double a2 = R[0][1];
	double a3 = R[0][2];

	double b1 = R[1][0];
	double b2 = R[1][1];
	double b3 = R[1][2];

	double c1 = R[2][0];
	double c2 = R[2][1];
	double c3 = R[2][2];

	double H = Zs - Z;

	double a11 = -f * cos(k) / H;
	double a12 = -f * sin(k) / H;

	double a13 = -x / H;
	double a23 = -y / H;
	
	double a14 = -(f + x * x / f) * cos(k) + x * y * sin(k) / f;
	double a15 = -(f + x * x / f) * sin(k) - x * y * cos(k) / f;
	double a24 = (f + y * y / f) * sin(k) - x * y * cos(k) / f;
	double a25 = -(f + y * y / f) * cos(k) - x * y * sin(k) / f;

	double a16 = y;
	double a26 = -x;
	
	double a21 = f * sin(k) / H;
	double a22 = -f * cos(k) / H;

	

	vector<vector<double>> A = creatmatrix(2, 6);
	A[0] = { a11,a12,a13,a14,a15,a16};
	A[1] = { a21,a22,a23,a24,a25,a26};

	// show_matrix(A);
	return A;
}
vector<vector<double>> to_l(double x, double y, double x_pre, double y_pre)
{
	vector<vector<double>> l = creatmatrix(2, 1);

	l[0] = { (x - x_pre) };
	l[1] = { (y - y_pre) };

	return l;
}

int main() 
{
	///////////////////////////////////////////////// 调试
	//vector<vector<double>> test = creatmatrix(3,3);
	//test[0] = { 1,2,3 };
	//test[1] = { 1,1,1 };
	//test[2] = { 2,1,2 };
	//cout << endl;
	//show_matrix(inverse(test));
	//cout << endl;
	//show_matrix(m_inverse(test));

	///////////////////////////////////////////////// 调试
	double lr = 1;

//基本数据
	// 内方位元素
	double f = 153.24e-3;
	double x0 = 0;
	double y0 = 0;
	double m = 50000;

	// 实验数据
	vector<vector<double>> x = creatmatrix(1, 4);
	vector<vector<double>> y = creatmatrix(1, 4);
	vector<vector<double>> X = creatmatrix(1, 4);
	vector<vector<double>> Y = creatmatrix(1, 4);
	vector<vector<double>> Z = creatmatrix(1, 4);

	x[0] = {-86.15e-3, -53.4e-3, -14.78e-3, 10.46e-3};
	y[0] = {-68.99e-3, 82.21e-3, -76.63e-3, 64.43e-3};
	X[0] = {36589.41, 37631.08, 39100.97, 40426.54};
	Y[0] = {25273.32, 31324.51, 24934.98, 30319.81};
	Z[0] = {2195.17, 728.69, 2386.5, 757.31};
	// 转置使其符合格式
	x = trans(x);
	y = trans(y);
	X = trans(X);
	Y = trans(Y);
	Z = trans(Z);

	// 角方位元素初值
	double p0 = 0;
	double w0 = 0;
	double k0 = 0;

	// 线方位元素初值
	double Xs0 = 0;
	double Ys0 = 0;
	double Zs0 = m*f;

	int i = 0;
	for (i = 0;i<X.size();i++)
	{
		Xs0 += X[i][0] / 4;
		Ys0 += Y[i][0] / 4;
		Zs0 += Z[i][0] / 4;
	}
	
// 开始计算
	// 计算预测值
vector<vector<double>> x_pre = to_x(X, Y, Z, p0, w0, k0, f, Xs0, Ys0, Zs0);
vector<vector<double>> y_pre = to_y(X, Y, Z, p0, w0, k0, f, Xs0, Ys0, Zs0);

	// 计算初始误差值
double error = 0;
for (int s = 0;s < x.size(); s++)
{
	error += abs(x[s][0] - x_pre[s][0]) / x.size();
	error += abs(y[s][0] - y_pre[s][0]) / y.size();
}
cout << endl << error;

vector<vector<double>> AA;
vector<vector<double>> ll;
vector<vector<double>> det;
vector<vector<double>> Naa;

// 迭代矫正外方位元素
int j = 0;
while ((j < 1000) )
{
	cout << endl << endl << "第" << j + 1 << "次迭代";

	// 小矩阵
	vector<vector<double>> A = creatmatrix(2, 6);
	vector<vector<double>> l = creatmatrix(2, 1);

	// 清空矩阵
	AA = creatmatrix((2 * X.size()), 6);
	ll = creatmatrix((2 * X.size()), 1);
	det = creatmatrix(6, 1);
	Naa = creatmatrix(6, 6);

	vector<vector<double>> W = creatmatrix(6, 1);

	for (int i = 0; i < X.size(); i++) 
	{
		// 计算每个点的矩阵
		A = to_A(X[i][0], Y[i][0], Z[i][0], p0, w0, k0, f, Xs0, Ys0, Zs0, x[i][0], y[i][0]);
		l = to_l(x[i][0], y[i][0], x_pre[i][0], y_pre[i][0]);

		// 填充大矩阵
		AA[2 * i] = A[0];
		AA[2 * i + 1] = A[1];
		ll[2 * i] = l[0];
		ll[2 * i + 1] = l[1];
	}

	// 计算误差方程
	Naa = multiply(trans(AA), AA);
	W = multiply(trans(AA), ll);

	// 判断行列式是否为0
	if (getA(Naa, Naa.size()) == 0)
	{
		cout << endl << "出现行列式为0情况，已经退出进程。";
		exit;
	}
	//show_matrix(m_inverse(Naa));
	det = multiply(m_inverse(Naa), W);
	//det = multiply_num(det, lr);
	//show_matrix(det);

	//更新参数
	Xs0 += det[0][0];
	Ys0 += det[1][0];
	Zs0 += det[2][0];
	p0 += det[3][0];
	w0 += det[4][0];
	k0 += det[5][0];

	//计算新误差
	x_pre = to_x(X, Y, Z, p0, w0, k0, f, Xs0, Ys0, Zs0);
	y_pre = to_y(X, Y, Z, p0, w0, k0, f, Xs0, Ys0, Zs0);
	error = 0;
	for (int s = 0;s < x.size(); s++)
	{
		error += abs(x[s][0] - x_pre[s][0]) / x.size();
		error += abs(y[s][0] - y_pre[s][0]) / y.size();
	}
	cout << endl << "当前平均误差值：" << error;


	j++;

	// 判断是否可以跳出循环
	//det = multiply_num(det, (1/lr));
	if (abs(det[0][0])<1e-6 && abs(det[1][0]) < 1e-6 && abs(det[2][0]) < 1e-6 && abs(det[3][0]) < 1e-3 && abs(det[4][0] )< 1e-3 && abs(det[5][0] )< 1e-3)
	{
		cout << endl <<endl<< "已达到精度要求";
		cout << endl << "当前改正值：" << endl;
		show_matrix(det);
		//cout << endl;
		//show_matrix(x);
		//cout << endl;
		//show_matrix(x_pre);

		cout << endl << "外方位元素:" << endl;
		cout << "Xs:" << Xs0 << endl;
		cout << "Ys:" << Ys0 << endl;
		cout << "Zs:" << Zs0 << endl;
		cout << "Phi:" << p0 << endl;
		cout << "Omgea:" << w0 << endl;
		cout << "Kapa:" << k0 << endl;
		break;
	}
}
// 计算精度
	// 计算V
vector<vector<double>> V = m_minus(multiply(AA, det), ll);
//show_matrix(V);

// 输出像点坐标和地面坐标
cout << endl << "像点坐标:";
for (i = 0;i < x_pre.size();i++)
{
	cout << endl << "x:" << x_pre[i][0] << " y:" << y_pre[i][0];
}
cout << endl;
cout << endl << "地面坐标:";
for (i = 0;i < x_pre.size();i++)
{
	cout << endl << "X:" << X[i][0] << " Y:" << Y[i][0] << " Z:" << Z[i][0];
}
cout << endl;


	// 计算单位权中误差 
vector<vector<double>> sigma_M = multiply_num(multiply(trans(V), V), 0.5);
double sigma0 = sqrt(sigma_M[0][0]);
cout << endl << "单位权中误差:" << sigma0 << endl;

	// 计算精度m
vector<double> mi = { -1,-1,-1,-1,-1,-1 };
cout << endl << "精度：";
for (int t = 0; t < 6; t++)
{
	mi[t] = sigma0 * sqrt( m_inverse(Naa)[t][t]);
	cout << endl << mi[t];
}
cout << endl;
return 0;
}