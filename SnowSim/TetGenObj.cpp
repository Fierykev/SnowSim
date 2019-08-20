#include "TetGenObj.h"

#include "CollisionDetection.h"

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include <igl/readOBJ.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>

#include <limits>

using namespace Eigen;

#define SCTP(A, B, C) A.dot(B.cross(C))

TetGenObj::TetGenObj(const char *filename)
{
	Eigen::MatrixXd mV;
	Eigen::MatrixXi mF;
	igl::readOBJ(filename, mV, mF);
	
	obj.Load(filename);

	{
		pV = new Eigen::MatrixX3d();
		pF = new Eigen::MatrixX3i();
		pT = new Eigen::MatrixX4i();
	}

	Eigen::MatrixX3d& V = *(Eigen::MatrixX3d*)pV;
	Eigen::MatrixX3i& F = *(Eigen::MatrixX3i*)pF;
	Eigen::MatrixX4i& T = *(Eigen::MatrixX4i*)pT;
	
	igl::copyleft::tetgen::tetrahedralize(
		mV,
		mF,
		"pq1.414a0.01",
		V,
		T,
		F);

	computeDistances();

	{
		// TODO: trash
		AABB = buildAABB(this);
	}
}

TetGenObj::~TetGenObj()
{
	delete pV;
	delete pF;
	delete pT;
}

void TetGenObj::Render()
{
	obj.Draw();
}

bool vertexPlaneDistanceLessThan(const Eigen::Vector3d &p,
	const Eigen::Vector3d &q0, const Eigen::Vector3d &q1, const Eigen::Vector3d &q2, double eta)
{
	Eigen::Vector3d c = (q1 - q0).cross(q2 - q0);
	return c.dot(p - q0)*c.dot(p - q0) < eta*eta*c.dot(c);
}

Eigen::Vector3d vertexFaceDistance(const Eigen::Vector3d &p,
	const Eigen::Vector3d &q0, const Eigen::Vector3d &q1, const Eigen::Vector3d &q2,
	double &q0bary, double &q1bary, double &q2bary)
{
	Eigen::Vector3d ab = q1 - q0;
	Eigen::Vector3d ac = q2 - q0;
	Eigen::Vector3d ap = p - q0;

	double d1 = ab.dot(ap);
	double d2 = ac.dot(ap);

	// corner and edge cases

	if (d1 <= 0 && d2 <= 0)
	{
		q0bary = 1.0;
		q1bary = 0.0;
		q2bary = 0.0;
		return q0 - p;
	}

	Eigen::Vector3d bp = p - q1;
	double d3 = ab.dot(bp);
	double d4 = ac.dot(bp);
	if (d3 >= 0 && d4 <= d3)
	{
		q0bary = 0.0;
		q1bary = 1.0;
		q2bary = 0.0;
		return q1 - p;
	}

	double vc = d1 * d4 - d3 * d2;
	if ((vc <= 0) && (d1 >= 0) && (d3 <= 0))
	{
		double v = d1 / (d1 - d3);
		q0bary = 1.0 - v;
		q1bary = v;
		q2bary = 0;
		return (q0 + v * ab) - p;
	}

	Eigen::Vector3d cp = p - q2;
	double d5 = ab.dot(cp);
	double d6 = ac.dot(cp);
	if (d6 >= 0 && d5 <= d6)
	{
		q0bary = 0;
		q1bary = 0;
		q2bary = 1.0;
		return q2 - p;
	}

	double vb = d5 * d2 - d1 * d6;
	if ((vb <= 0) && (d2 >= 0) && (d6 <= 0))
	{
		double w = d2 / (d2 - d6);
		q0bary = 1 - w;
		q1bary = 0;
		q2bary = w;
		return (q0 + w * ac) - p;
	}

	double va = d3 * d6 - d5 * d4;
	if ((va <= 0) && (d4 - d3 >= 0) && (d5 - d6 >= 0))
	{
		double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		q0bary = 0;
		q1bary = 1.0 - w;
		q2bary = w;

		return (q1 + w * (q2 - q1)) - p;
	}

	// face case
	double denom = 1.0 / (va + vb + vc);
	double v = vb * denom;
	double w = vc * denom;
	double u = 1.0 - v - w;
	q0bary = u;
	q1bary = v;
	q2bary = w;
	return (u*q0 + v * q1 + w * q2) - p;
}

void TetGenObj::computeDistances()
{
	Eigen::MatrixX3d& V = *(Eigen::MatrixX3d*)pV;
	Eigen::MatrixX3i& F = *(Eigen::MatrixX3i*)pF;
	Eigen::MatrixX4i& T = *(Eigen::MatrixX4i*)pT;

	int nverts = (int)V.rows();
	int nfaces = (int)F.rows();
	distances.resize(nverts);

#pragma omp parallel for
	for (int i = 0; i < nverts; i++)
	{
		double dist = std::numeric_limits<double>::infinity();

#pragma omp parallel for
		for (int j = 0; j < nfaces; j++)
		{
			double dummy0, dummy1, dummy2;
			if (vertexPlaneDistanceLessThan(V.row(i), V.row(F(j, 0)), V.row(F(j, 1)), V.row(F(j, 2)), dist))
			{
				Vector3d distvec = vertexFaceDistance(V.row(i), V.row(F(j, 0)), V.row(F(j, 1)), V.row(F(j, 2)), dummy0, dummy1, dummy2);
				dist = min(dist, distvec.norm());
			}
		}
		distances[i] = dist;
	}
}

double TetGenObj::distance(float3 pFace, int tet) const
{
	Eigen::MatrixX3d& V = *(Eigen::MatrixX3d*)pV;
	Eigen::MatrixX3i& F = *(Eigen::MatrixX3i*)pF;
	Eigen::MatrixX4i& T = *(Eigen::MatrixX4i*)pT;

	Vector3d p = { pFace.x, pFace.y, pFace.z };

	// TODO compute distance from point to object boundary
	Vector4i tetro = T.row(tet);

	Vector3d v[4];
	Vector4d dist;
	{
		for (int i = 0; i < 4; i++)
		{
			v[i] = V.row(tetro[i]);
			dist[i] = distances[tetro[i]];
		}
	}

	Vector3d a, b, c, d;
	{
		a = v[0];
		b = v[1];
		c = v[2];
		d = v[3];
	}

	Vector3d ap = p - a;
	Vector3d bp = p - b;

	Vector3d ab = b - a;
	Vector3d ac = c - a;
	Vector3d ad = d - a;

	Vector3d bc = c - b;
	Vector3d bd = d - b;

	// Standard calculation.
	Vector4d weight;
	{
		weight[0] = SCTP(bp, bd, bc);
		weight[1] = SCTP(ap, ac, ad);
		weight[2] = SCTP(ap, ad, ab);
		weight[3] = SCTP(ap, ab, ac);

		weight *= 1.0 / SCTP(ab, ac, ad);
	}

	return -dist.dot(weight);
}

float3 TetGenObj::Ddistance(int tet) const
{
	Eigen::MatrixX3d& V = *(Eigen::MatrixX3d*)pV;
	Eigen::MatrixX3i& F = *(Eigen::MatrixX3i*)pF;
	Eigen::MatrixX4i& T = *(Eigen::MatrixX4i*)pT;

	//TODO: compute derivative of distance from point to boundary
	Vector3d result(0, 0, 0);

	Vector4i tetro = T.row(tet);

	Vector3d v[4];
	Vector4d dist;
	{
		for (int i = 0; i < 4; i++)
		{
			v[i] = V.row(tetro[i]);
			dist[i] = distances[tetro[i]];
		}
	}

	Vector3d a, b, c, d;
	{
		a = v[0];
		b = v[1];
		c = v[2];
		d = v[3];
	}

	Vector3d ab = b - a;
	Vector3d ac = c - a;
	Vector3d ad = d - a;

	Vector3d bc = c - b;
	Vector3d bd = d - b;

	// Standard calculation.
	MatrixXd weight(4, 3);
	{
		auto div = 1.0 / SCTP(ab, ac, ad);

		weight.row(0) = bd.cross(bc) * div;
		weight.row(1) = ac.cross(ad) * div;
		weight.row(2) = ad.cross(ab) * div;
		weight.row(3) = ab.cross(ac) * div;
	}
	result = dist.transpose() * weight;

	return make_float3(-result.x(), -result.y(), -result.z());
}