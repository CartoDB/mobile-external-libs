#ifndef _CGLIB_RAY_H
#define _CGLIB_RAY_H

#include "vec.h"
#include "mat.h"
#include "bbox.h"
#include "frustum3.h"

#include <algorithm>
#include <utility>

namespace cglib
{
    
    /**
     * N-dimensional ray
     */
    
    template <typename T, size_t N, typename Traits = float_traits<T> >
        class ray
    {
        
    public:

        typedef T value_type;
        typedef Traits traits_type;

        constexpr ray() = default;
        
        constexpr explicit ray(const vec<T, N, Traits> & origin, const vec<T, N, Traits> & direction) : origin(origin), direction(direction) { }

        constexpr vec<T, N, Traits> operator () (T t) const
        {
            return origin + direction * t;
        }
        
        template <typename S, typename TraitsS>
            static ray<T, N, Traits> convert(const ray<S, N, TraitsS> & r)
        {
            return ray(vec<T, N, Traits>::convert(r.origin), vec<T, N, Traits>::convert(r.direction));
        }
        
    public:

        vec<T, N, Traits> origin;
        vec<T, N, Traits> direction;
    };

    /**
     * Operators
     */

    template <typename T, size_t N, typename Traits> CGLIB_FORCEINLINE
        bool operator == (const ray<T, N, Traits> & r1, const ray<T, N, Traits> & r2)
    {
        return r1.origin == r2.direction && r1.direction == r2.direction;
    }

    template <typename T, size_t N, typename Traits> CGLIB_FORCEINLINE
        bool operator != (const ray<T, N, Traits> & r1, const ray<T, N, Traits> & r2)
    {
        return !(r1 == r2);
    }

    template <typename T, size_t N, typename Traits>
        bool intersect_plane(const vec<T, N+1, Traits> & plane, const ray<T, N, Traits> & r, T * t_ptr = nullptr)
    {
        vec<T, N, Traits> n = proj_o(plane);
        T nd = dot_product(n, r.direction);
        if (nd == 0)
            return false;

        T t = -(dot_product(r.origin, n) + plane(N)) / nd;
        if (t < 0)
            return false;

        if (t_ptr)
            *t_ptr = t;
        return true;
    }

    template <typename T, typename Traits>
        bool intersect_triangle(const vec3<T, Traits> & p0, const vec3<T, Traits> & p1, const vec3<T, Traits> & p2, const ray<T, 3, Traits> & r, T * t_ptr = nullptr)
    {
        vec3<T, Traits> u = p1 - p0;
        vec3<T, Traits> v = p2 - p0;
        vec3<T, Traits> n = vector_product(u, v);
        if (norm(n) == 0)
            return false;
        n = unit(n);

        T nd = dot_product(n, r.direction);
        if (nd == 0)
            return false;

        T d = dot_product(p0, n);
        T t = (d - dot_product(r.origin, n)) / nd;
        if (t < 0)
            return false;
        
        vec3<T, Traits> w = r(t) - p0;
        T uu = dot_product(u, u);
        T uv = dot_product(u, v);
        T vv = dot_product(v, v);
        T wu = dot_product(w, u);
        T wv = dot_product(w, v);

        T det = uv * uv - uu * vv;

        T s0 = (uv * wv - vv * wu) / det;
        T s1 = (uv * wu - uu * wv) / det;

        if (!(s0 >= 0 && s1 >= 0 && s0 + s1 <= 1))
            return false;

        if (t_ptr)
            *t_ptr = t;
        return true;
    }

    template <typename T, size_t N, typename Traits>
        bool intersect_bbox(const bbox<T, N, Traits> & b, const ray<T, N, Traits> & r, T * t_ptr = nullptr)
    {
        T tmin = -Traits::infinity();
        T tmax =  Traits::infinity();
        for (size_t i = 0; i < N; i++)
        {
            if (r.direction(i) != 0)
            {
                T t1 = (b.min(i) - r.origin(i)) / r.direction(i);
                T t2 = (b.max(i) - r.origin(i)) / r.direction(i);
                tmin = std::max(tmin, std::min(t1, t2));
                tmax = std::min(tmax, std::max(t1, t2));
            }
        }

        if (tmax < 0 || tmin > tmax)
            return false;

        if (t_ptr)
            *t_ptr = tmin < 0 ? tmax : tmin;
        return true;
    }

    template <typename T, typename Traits>
        bool intersect_frustum(const frustum3<T, Traits> & fru, const ray<T, 3, Traits> & r, T * t_ptr = nullptr)
    {
        T tmin = -Traits::infinity();
        T tmax =  Traits::infinity();
        for (size_t i = 0; i < fru.planes.size(); i++)
        {
            T t = 0;
            if (intersect_plane(fru.planes[i], r, &t))
            {
                vec<T, 3, Traits> n = proj_o(fru.planes[i]);
                if (dot_product(n, r.direction) > 0)
                {
                    tmin = std::min(tmin, t);
                }
                else
                {
                    tmax = std::max(tmax, t);
                }
            }
        }

        if (tmax < 0 || tmin > tmax)
            return false;

        if (t_ptr)
            *t_ptr = tmin < 0 ? tmax : tmin;
        return true;
    }

    template <typename T, size_t N, typename Traits>
        ray<T, N, Traits> transform_ray(const ray<T, N, Traits> & r, const mat<T, N+1, Traits> & m)
    {
        vec<T, N, Traits> origin = transform_point(r.origin, m);
        vec<T, N, Traits> direction = transform_point(r.origin + r.direction, m) - origin;
        return ray<T, N, Traits>(origin, direction);
    }

    /**
     * Reads ray from stream.
     * @relates ray
     */
    
    template <typename T, size_t N, typename Traits, typename CharT, typename CharTraits> std::basic_istream<CharT, CharTraits> &
        operator >> (std::basic_istream<CharT, CharTraits> & is, ray<T, N, Traits> & r)
    {
        is >> r.origin;
        CharT ch;
        is >> ch;
        if (ch != ',')
        {
            is.setstate(std::ios_base::failbit);
            return is;
        }
        is >> r.direction;
        return is;
    }

    /**
     * Writes ray info to stream.
     * @relates ray
     */
    
    template <typename T, size_t N, typename Traits, typename CharT, typename CharTraits> std::basic_ostream<CharT, CharTraits> &
        operator << (std::basic_ostream<CharT, CharTraits> & os, const ray<T, N, Traits> & r)
    {
        os << r.origin << ',' << r.direction;
        return os;
    }
    
    /**
     * Commonly used instances for 2D, 3D and 4D cases
     */
    
    template <typename T, typename Traits = float_traits<T> > using ray2 = ray<T, 2, Traits>;
    template <typename T, typename Traits = float_traits<T> > using ray3 = ray<T, 3, Traits>;
    template <typename T, typename Traits = float_traits<T> > using ray4 = ray<T, 4, Traits>;

}

#endif
