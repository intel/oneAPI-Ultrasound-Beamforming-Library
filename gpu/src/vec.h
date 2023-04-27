// ================================================================================================
//
// If not explicitly stated: Copyright (C) 2016, all rights reserved,
//      Rüdiger Göbl
//		Email r.goebl@tum.de
//      Chair for Computer Aided Medical Procedures
//      Technische Universität München
//      Boltzmannstr. 3, 85748 Garching b. München, Germany
//
// ================================================================================================

#ifndef __VEC_H__
#define __VEC_H__

#include <sycl/sycl.hpp>
#include <cmath>
#include <dpct/dpct.hpp>

#ifndef M_PI
/// Definition of pi for the cuda compile path, as cuda math.h does not seem to
/// provide it.
constexpr auto M_PI = 3.14159265358979323846;
#endif  //! M_PI
/// Definition of eps following the value of matlabs "eps()"
constexpr double M_EPS = 2.2204e-16;
template <typename ElementType>
struct vec2T;
template <typename ElementType>
struct vec3T;
template <typename ElementType>
struct vec4T;

/// Vector of two elements of type ElementType
template <typename ElementType>
struct vec2T {
  /// The first element of the two-vector
  ElementType x;
  /// The second element of the two-vector
  ElementType y;

  /// Explicit numeric conversion operator. Casts both elements seperately to
  /// NewElementType
  template <typename NewElementType>
  explicit operator vec2T<NewElementType>() const {
    return vec2T<NewElementType>{static_cast<NewElementType>(this->x),
                                 static_cast<NewElementType>(this->y)};
  }

  /// Equality operator. Compares this and b elementwise for equality
  bool operator==(const vec2T<ElementType>& b) const {
    return x == b.x && y == b.y;
  }

  /// Inequality operator. Compares this and b elementwise for inequality
  bool operator!=(const vec2T<ElementType>& b) const {
    return x != b.x || y != b.y;
  }
};

/// Vector of three elements of type ElementType
template <typename ElementType>
struct vec3T {
  /// The first element of the three-vector
  ElementType x;
  /// The second element of the three-vector
  ElementType y;
  /// The third element of the three-vector
  ElementType z;

  /// Conversion of a 3-vector representing a POINT to homogeneous coordinates
  vec4T<ElementType> pointToHom() const {
    return vec4T<ElementType>({this->x, this->y, this->z, 1});
  }
  /// Explicit conversion of a 3-vector representing a POINT to homogeneous
  /// coordinates
  explicit operator vec4T<ElementType>() const { return pointToHom(); };

  /// Conversion of a 3-vector representing a VECTOR to homogeneous coordinates
  vec4T<ElementType> vectorToHom() const {
    return vec4T<ElementType>({this->x, this->y, this->z, 0});
  }

  /// Explicit numeric conversion operator. Casts both elements seperately to
  /// NewElementType
  template <typename NewElementType>
  explicit operator vec3T<NewElementType>() const {
    return vec3T<NewElementType>{static_cast<NewElementType>(this->x),
                                 static_cast<NewElementType>(this->y),
                                 static_cast<NewElementType>(this->z)};
  }

  /// Equality operator. Compares this and b elementwise for equality
  bool operator==(const vec3T<ElementType>& b) const {
    return x == b.x && y == b.y && z == b.z;
  }

  /// Inequality operator. Compares this and b elementwise for inequality
  bool operator!=(const vec3T<ElementType>& b) const {
    return x != b.x || y != b.y || z != b.z;
  }
};

/// Vector of four elements of type ElementType
template <typename ElementType>
struct vec4T {
  /// The first element of the four-vector
  ElementType x;
  /// The second element of the four-vector
  ElementType y;
  /// The third element of the four-vector
  ElementType z;
  /// The fourth element of the four-vector
  ElementType w;
};

/// Rectangle in 2D of type ElementType
template <typename ElementType>
struct rect2T {
  /// Inclusive begin of the rectangle in top-left
  vec2T<ElementType> begin;
  /// Inclusive end of the rectangle in bottom-right
  vec2T<ElementType> end;
};

/// Single precision two-vector
typedef vec2T<float> vec2f;
/// Single precision three-vector
typedef vec3T<float> vec3f;
/// Single precision four-vector
typedef vec4T<float> vec4f;

/// Double precision two-vector
typedef vec2T<double> vec2d;
/// Double precision three-vector
typedef vec3T<double> vec3d;
/// Double precision four-vector
typedef vec4T<double> vec4d;

/// Integer two-vector
typedef vec2T<int> vec2i;
/// Integer three-vector
typedef vec3T<int> vec3i;
/// Integer four-vector
typedef vec4T<int> vec4i;

/// size_t two-vector
typedef vec2T<size_t> vec2s;
/// size_t three-vector
typedef vec3T<size_t> vec3s;
/// size_t four-vector
typedef vec4T<size_t> vec4s;

/// Double precision two-vector
typedef vec2d vec2;
/// Double precision three-vector
typedef vec3d vec3;
/// Double precision four-vector
typedef vec4d vec4;

/// Double precision three-vector
typedef vec3 vec;

/// Single precision rectangle
typedef rect2T<float> rect2f;
/// Double precision rectangle
typedef rect2T<double> rect2d;
/// Integer rectangle
typedef rect2T<int> rect2i;
/// size_t rectangle
typedef rect2T<size_t> rect2s;

/// Double precision rectangle
typedef rect2d rect2;
/// Double precision rectangle
typedef rect2 rect;

/// Element-wise sum of a two-vector and a scalar
template <typename Ta, typename Tb>
SYCL_EXTERNAL inline vec2T<Ta> operator+(const vec2T<Ta>& a, const Tb& b) {
  return vec2T<Ta>({a.x + b, a.y + b});
}
/// Negation of a two-vector
template <typename Ta>
SYCL_EXTERNAL inline vec2T<Ta> operator-(const vec2T<Ta>& a) {
  return vec2T<Ta>({-a.x, -a.y});
}
/// Element-wise subtraction of a two-vector and a scalar
template <typename Ta, typename Tb>
SYCL_EXTERNAL inline vec2T<Ta> operator-(const vec2T<Ta>& a, const Tb& b) {
  return vec2T<Ta>({a.x - b, a.y - b});
}
/// Product of a two-vector and a scalar
template <typename Ta, typename Tb>
SYCL_EXTERNAL inline vec2T<Ta> operator*(const vec2T<Ta>& a, const Tb& b) {
  return vec2T<Ta>({a.x * b, a.y * b});
}
/// Element-wise division of a two-vector and a scalar
template <typename Ta, typename Tb>
SYCL_EXTERNAL inline vec2T<Ta> operator/(const vec2T<Ta>& a, const Tb& b) {
  return vec2T<Ta>({a.x / b, a.y / b});
}
/// Element-wise sum of a scalar and a two-vector
template <typename Ta, typename Tb>
SYCL_EXTERNAL inline vec2T<Ta> operator+(const Tb& b, const vec2T<Ta>& a) {
  return vec2T<Ta>({a.x + b, a.y + b});
}
/// Element-wise difference of a scalar and a two-vector
template <typename Ta, typename Tb>
SYCL_EXTERNAL inline vec2T<Ta> operator-(const Tb& b, const vec2T<Ta>& a) {
  return vec2T<Ta>({b - a.x, b - a.y});
}
/// Product of a scalar and a two-vector
template <typename Ta, typename Tb>
SYCL_EXTERNAL inline vec2T<Ta> operator*(const Tb& b, const vec2T<Ta>& a) {
  return vec2T<Ta>({a.x * b, a.y * b});
}
/// Element-wise division of two-vectors
template <typename Ta, typename Tb>
SYCL_EXTERNAL inline vec2T<Ta> operator/(const vec2T<Ta>& a,
                                         const vec2T<Tb>& b) {
  return vec2T<Ta>({a.x / b.x, a.y / b.y});
}
/// Sum of two-vectors
template <typename Ta, typename Tb>
SYCL_EXTERNAL inline vec2T<Ta> operator+(const vec2T<Ta>& a,
                                         const vec2T<Tb>& b) {
  return vec2T<Ta>({a.x + b.x, a.y + b.y});
}
/// Difference of two-vectors
template <typename Ta, typename Tb>
SYCL_EXTERNAL inline vec2T<Ta> operator-(const vec2T<Ta>& a,
                                         const vec2T<Tb>& b) {
  return vec2T<Ta>({a.x - b.x, a.y - b.y});
}
/// Element-wise product of two-vectors
template <typename Ta, typename Tb>
SYCL_EXTERNAL inline vec2T<Ta> operator*(const vec2T<Ta>& a,
                                         const vec2T<Tb>& b) {
  return vec2T<Ta>({a.x * b.x, a.y * b.y});
}
/// Euclidean norm of a two-vector
template <typename Ta>
inline Ta norm(const vec2T<Ta>& a) {
  return sycl::sqrt(a.x * a.x + a.y * a.y);
}
/// Normalization of a two-vector
template <typename Ta>
inline vec2T<Ta> normalize(const vec2T<Ta>& a) {
  return a / norm(a);
}

/// Element-wise round of a two-vector
template <typename Ta>
inline vec2T<Ta> round(const vec2T<Ta>& a) {
  return {round(a.x), round(a.y)};
}
/// Element-wise floor of a two-vector
template <typename Ta>
inline vec2T<Ta> floor(const vec2T<Ta>& a) {
  return {sycl::floor((double)(a.x)), sycl::floor((double)(a.y))};
}
/// Element-wise ceil of a two-vector
template <typename Ta>
inline vec2T<Ta> ceil(const vec2T<Ta>& a) {
  return {sycl::ceil((double)(a.x)), sycl::ceil((double)(a.y))};
}
/// Element-wise minimum of two-vectors
template <typename Ta>
inline vec2T<Ta> min(const vec2T<Ta>& a, const vec2T<Ta>& b) {
  return {std::min(a.x, b.x), std::min(a.y, b.y)};
}
/// Element-wise maximum of two-vectors
template <typename Ta>
inline vec2T<Ta> max(const vec2T<Ta>& a, const vec2T<Ta>& b) {
  return {std::max(a.x, b.x), std::max(a.y, b.y)};
}

/// Element-wise sum of a three-vector and a scalar
template <typename Ta, typename Tb>
SYCL_EXTERNAL inline vec3T<Ta> operator+(const vec3T<Ta>& a, const Tb& b) {
  return vec3T<Ta>({a.x + b, a.y + b, a.z + b});
}
/// Negation of a three-vector
template <typename Ta>
SYCL_EXTERNAL inline vec3T<Ta> operator-(const vec3T<Ta>& a) {
  return vec3T<Ta>({-a.x, -a.y, -a.z});
}
/// Element-wise subtraction of a three-vector and a scalar
template <typename Ta, typename Tb>
SYCL_EXTERNAL inline vec3T<Ta> operator-(const vec3T<Ta>& a, const Tb& b) {
  return vec3T<Ta>({a.x - b, a.y - b, a.z - b});
}
/// Product of a three-vector and a scalar
template <typename Ta, typename Tb>
SYCL_EXTERNAL inline vec3T<Ta> operator*(const vec3T<Ta>& a, const Tb& b) {
  return vec3T<Ta>({a.x * b, a.y * b, a.z * b});
}
/// Element-wise division of a three-vector and a scalar
template <typename Ta, typename Tb>
SYCL_EXTERNAL inline vec3T<Ta> operator/(const vec3T<Ta>& a, const Tb& b) {
  return vec3T<Ta>({a.x / b, a.y / b, a.z / b});
}
/// Element-wise sum of a scalar and a three-vector
template <typename Ta, typename Tb>
SYCL_EXTERNAL inline vec3T<Ta> operator+(const Tb& b, const vec3T<Ta>& a) {
  return vec3T<Ta>({a.x + b, a.y + b, a.z + b});
}
/// Element-wise difference of a scalar and a three-vector
template <typename Ta, typename Tb>
SYCL_EXTERNAL inline vec3T<Ta> operator-(const Tb& b, const vec3T<Ta>& a) {
  return vec3T<Ta>({b - a.x, b - a.y, b - a.z});
}
/// Product of a scalar and a three-vector
template <typename Ta, typename Tb>
SYCL_EXTERNAL inline vec3T<Ta> operator*(const Tb& b, const vec3T<Ta>& a) {
  return vec3T<Ta>({a.x * b, a.y * b, a.z * b});
}
/// Element-wise division of three-vectors
template <typename Ta, typename Tb>
SYCL_EXTERNAL inline vec3T<Ta> operator/(const vec3T<Ta>& a,
                                         const vec3T<Tb>& b) {
  return vec3T<Ta>({a.x / b.x, a.y / b.y, a.z / b.z});
}
/// Sum of three-vectors
template <typename Ta, typename Tb>
SYCL_EXTERNAL inline vec3T<Ta> operator+(const vec3T<Ta>& a,
                                         const vec3T<Tb>& b) {
  return vec3T<Ta>({a.x + b.x, a.y + b.y, a.z + b.z});
}
/// Difference of three-vectors
template <typename Ta, typename Tb>
SYCL_EXTERNAL inline vec3T<Ta> operator-(const vec3T<Ta>& a,
                                         const vec3T<Tb>& b) {
  return vec3T<Ta>({a.x - b.x, a.y - b.y, a.z - b.z});
}
/// Element-wise product of three-vectors
template <typename Ta, typename Tb>
SYCL_EXTERNAL inline vec3T<Ta> operator*(const vec3T<Ta>& a,
                                         const vec3T<Tb>& b) {
  return vec3T<Ta>({a.x * b.x, a.y * b.y, a.z * b.z});
}

/// Euclidean norm of a three-vector
template <typename Ta>
inline Ta norm(const vec3T<Ta>& a) {
  return sycl::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}
/// Normalization of a three-vector
template <typename Ta>
inline vec3T<Ta> normalize(const vec3T<Ta>& a) {
  return a / norm(a);
}
/// Dot-product of two three-vectors
template <typename Ta, typename Tb>
inline typename std::common_type<Ta, Tb>::type dot(const vec3T<Ta>& a,
                                                   const vec3T<Tb>& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
/// Cross-product of two three-vectors
template <typename Ta, typename Tb>
inline vec3T<typename std::common_type<Ta, Tb>::type> cross(
    const vec3T<Ta>& a, const vec3T<Tb>& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
/// Determinant of matric \f$[a, b, c]\f$
template <typename Ta, typename Tb, typename Tc>
inline typename std::common_type<Ta, Tb, Tc>::type det(const vec3T<Ta>& a,
                                                       const vec3T<Tb>& b,
                                                       const vec3T<Tc>& c) {
  return sycl::fabs(dot(a, cross(b, c)));
}
/// Element-wise round of a three-vector
template <typename Ta>
inline vec3T<Ta> round(const vec3T<Ta>& a) {
  return {round(a.x), round(a.y), round(a.z)};
}
/// Element-wise floor of a three-vector
template <typename Ta>
inline vec3T<Ta> floor(const vec3T<Ta>& a) {
  return {sycl::floor((double)(a.x)), sycl::floor((double)(a.y)),
          sycl::floor((double)(a.z))};
}
/// Element-wise ceil of a three-vector
template <typename Ta>
inline vec3T<Ta> ceil(const vec3T<Ta>& a) {
  return {sycl::ceil((double)(a.x)), sycl::ceil((double)(a.y)),
          sycl::ceil((double)(a.z))};
}

/// Element-wise minimum of three-vectors
template <typename Ta>
inline vec3T<Ta> min(const vec3T<Ta>& a, const vec3T<Ta>& b) {
  return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)};
}

/// Element-wise maximum of three-vectors
template <typename Ta>
inline vec3T<Ta> max(const vec3T<Ta>& a, const vec3T<Ta>& b) {
  return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};
}

/// Spherical Linear Interpolation (SLERP) of two three-vectors
template <typename T>
vec3T<T> inline slerp3(const vec3T<T>& a, const vec3T<T>& b, const T& t) {
  T omega = sycl::acos((double)(dot(a, b)));
  if (omega < M_EPS) {
    return a;
  }
  vec3T<T> ret;
  if (t < M_EPS) {
    ret = a;
  } else if ((T)1.0 - t < M_EPS) {
    ret = b;
  } else {
    ret = (sycl::sin(((T)1.0 - t) * omega) / sycl::sin(omega)) * a +
          (sycl::sin(t * omega) / sycl::sin(omega)) * b;
  }
  return ret;
}

#endif  //!__VEC_H__
