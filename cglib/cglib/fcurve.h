#ifndef _CGLIB_FCURVE_H
#define _CGLIB_FCURVE_H

#include "base.h"
#include "vec.h"

#include <cstdlib>
#include <vector>

namespace cglib
{

    /**
     * Curve type.
     */

    enum class fcurve_type
    {
        step,
        linear,
        cubic
    };

    /**
     * Curve class that support stepwise, linear and cubic curves.
     * Parametrized by value type, dimension and value type traits.
     */

    template <typename T, size_t N, typename Traits = float_traits<T> >
        class fcurve
    {

    public:

        typedef T scalar_type;
        typedef cglib::vec<T, N, Traits> value_type;

        struct key_type
        {
            value_type pos, dpos_left, dpos_right;

            key_type() : pos(value_type::zero()), dpos_left(value_type::zero()), dpos_right(value_type::zero()) { }
            explicit key_type(const value_type & pos, const value_type & dpos_left = value_type::zero(), const value_type & dpos_right = value_type::zero()) : pos(pos), dpos_left(dpos_left), dpos_right(dpos_right) { }

            void compute_tangents(const key_type * prev, const key_type * next)
            {
                value_type diff = value_type::zero();
                if (prev && next)
                    diff = next->pos - prev->pos;
                scalar_type dl = 0;
                if (prev)
                    dl = -(pos(0) - prev->pos(0));
                dpos_left = diff * dl;
                dpos_left(0) = dl / 3;
                scalar_type dr = 0;
                if (next)
                    dr = next->pos(0) - pos(0);
                dpos_right = diff * dr;
                dpos_right(0) = dr / 3;
            }
        };

        fcurve() : _type(fcurve_type::linear), _keys() { }
        explicit fcurve(fcurve_type ty) : _type(ty), _keys() { }

        fcurve_type type() const
        {
            return _type;
        }

        void clear()
        {
            _keys.clear();
        }

        size_t size() const
        {
            return _keys.size();
        }

        key_type & at(size_t n)
        {
            return _keys.at(n);
        }

        const key_type & at(size_t n) const
        {
            return _keys.at(n);
        }

        void remove(size_t n)
        {
            if (n < _keys.size())
                _keys.erase(_keys.begin() + n);
        }

        size_t insert(const key_type & k)
        {
            size_t ir = find_right(k.pos(0));

            // Insert after last key?
            if (ir >= _keys.size())
            {
                _keys.push_back(k);
            }
            else
            {
                _keys.insert(_keys.begin() + ir, k);
            }
            return ir;
        }

        size_t find_right(scalar_type t) const
        {
            // Handle border cases
            if (_keys.empty())
                return 0;
            if (t < _keys.front().pos(0))
                return 0;
            if (t >= _keys.back().pos(0))
                return _keys.size();

            // Use binary search to find appropriate key
            size_t il = 0;
            size_t ir = _keys.size() - 1;
            while (true)
            {
                size_t im = (il + ir) / 2;
                const key_type & km = _keys[im];
                if (t > km.pos(0))
                {
                    if (il == im)
                        break;
                    il = im;
                }
                else
                {
                    if (ir == im)
                        break;
                    ir = im;
                }
            }
            return il + 1;
        }

        value_type evaluate(scalar_type t) const
        {
            constexpr int iterations = 20;
            constexpr scalar_type epsilon = float_traits<scalar_type>::epsilon();

            // Find right index
            size_t ir = find_right(t);

            // Check if evaluation proceeds last key
            if (ir >= _keys.size())
            {
                if (ir == 0)
                    return value_type();
                const key_type & k = _keys.back();
                if (_type == fcurve_type::step || k.dpos_right(0) <= epsilon)
                    return k.pos;
                scalar_type dt = t - k.pos(0);
                return k.pos + k.dpos_right * (dt / k.dpos_right(0));
            }

            // Check if evaluation time preceeds first key
            if (ir == 0)
            {
                const key_type & k = _keys.front();
                if (_type == fcurve_type::step || -k.dpos_left(0) <= epsilon)
                    return k.pos;
                scalar_type dt = k.pos(0) - t;
                return k.pos + k.dpos_left * (dt / k.dpos_left(0));
            }

            // Get left and right keys
            const key_type & kl = _keys.at(ir - 1);
            const key_type & kr = _keys.at(ir - 0);

            // Step/Linear fcurve?
            if (_type == fcurve_type::step)
            {
                return kl.pos;
            }
            else if (_type == fcurve_type::linear)
            {
                value_type dpos = kr.pos - kl.pos;
                if (dpos(0) <= epsilon)
                    return kl.pos;
                scalar_type dt = t - kl.pos(0);
                return kl.pos + dpos * (dt / dpos(0));
            }

            // Create 2d Hermite curve f and find y (given t) such that (y, t) = f(s).
            // We can assume that the second argument is monotonically increasing.
            scalar_type tl = kl.pos(0);
            scalar_type tr = kr.pos(0);
            for (int i = 0; i < iterations; i++)
            {
                scalar_type tm = (tl + tr) / 2;
                scalar_type tt = eval_hermite(kl, kr, tm)(0);
                if (t > tt)
                    tl = tm;
                else
                    tr = tm;
            }

            // Perform evaluation
            return eval_hermite(kl, kr, (tl + tr) / 2);
        }

        template <typename It>
            static fcurve<T, N, Traits> create(fcurve_type ty, It begin, It end)
        {
            fcurve<T, N, Traits> curve(ty);

            // Build the curve by inserting each key frame
            for (It it = begin; it != end; it++)
            {
                curve.insert(key_type(*it));
            }

            // Calculate tangents, if cubic curve
            if (ty == fcurve_type::cubic)
            {
                for (size_t i = 0; i < curve._keys.size(); i++)
                {
                    key_type * prev = (i > 0 ? &curve._keys[i - 1] : nullptr);
                    key_type * next = (i + 1 < curve._keys.size() ? &curve._keys[i + 1] : nullptr);
                    curve._keys[i].compute_tangents(prev, next);
                }
            }

            return curve;
        }

    protected:

        value_type eval_hermite(const key_type & kl, const key_type & kr, scalar_type t) const
        {
            scalar_type s1 = (t - kl.pos(0)) / (kr.pos(0) - kl.pos(0));
            scalar_type s2 = s1 * s1;
            scalar_type s3 = s2 * s1;
            scalar_type h1 = 2 * s3 - 3 * s2 + 1;
            scalar_type h2 = 1 - h1;
            scalar_type h3 = (s3 - 2 * s2 + s1) * 3;
            scalar_type h4 = (s3 - s2) * (-3);
            return kl.pos * h1 + kr.pos * h2 + kl.dpos_right * h3 + kr.dpos_left * h4;
        }

        fcurve_type _type;
        std::vector<key_type> _keys; // sorted array of keys
    };

    /**
     * @relates fcurve
     */
    
    template <typename T, size_t N, typename Traits> bool
        operator == (const fcurve<T, N, Traits> & curve1, const fcurve<T, N, Traits> & curve2)
    {
        if (curve1.type() != curve2.type() || curve1.size() != curve2.size())
            return false;
        for (size_t i = 0; i < curve1.size(); i++)
        {
            const typename fcurve<T, N, Traits>::key_type & key1 = curve1.at(i);
            const typename fcurve<T, N, Traits>::key_type & key2 = curve2.at(i);
            if (key1.pos != key2.pos || key1.dpos_left != key2.dpos_left || key1.dpos_right != key2.dpos_right)
                return false;
        }
        return true;
    }
    
    /**
     * @relates fcurve
     */
    
    template <typename T, size_t N, typename Traits> bool
        operator != (const fcurve<T, N, Traits> & curve1, const fcurve<T, N, Traits> & curve2)
    {
        return !(curve1 == curve2);
    }

    /**
     * Reads fcurve info from stream.
     * @relates fcurve
     */

    template <typename T, size_t N, typename Traits, typename CharT, typename CharTraits> std::basic_istream<CharT, CharTraits> &
        operator >> (std::basic_istream<CharT, CharTraits> & is, fcurve<T, N, Traits> & curve)
    {
        CharT ch;
        is >> ch;
        if (ch != '[')
        {
            is.setstate(std::ios_base::failbit);
            return is;
        }

        char type;
        is >> type;
        if (type == 's')
        {
            curve = fcurve<T, N, Traits>(fcurve_type::step);
        }
        else if (type == 'l')
        {
            curve = fcurve<T, N, Traits>(fcurve_type::linear);
        }
        else if (type == 'c')
        {
            curve = fcurve<T, N, Traits>(fcurve_type::cubic);
        }
        else
        {
            is.setstate(std::ios_base::failbit);
            return is;
        }

        is >> ch;
        if (ch != ';')
        {
            is.setstate(std::ios_base::failbit);
            return is;
        }

        do
        {
            typename fcurve<T, N, Traits>::key_type key;
            is >> key.pos;
            is >> ch;
            if (ch != ',')
            {
                is.setstate(std::ios_base::failbit);
                break;
            }
            is >> key.dpos_left;
            is >> ch;
            if (ch != ',')
            {
                is.setstate(std::ios_base::failbit);
                break;
            }
            is >> key.dpos_right;
            curve.insert(key);
            is >> ch;
            if (ch != ']' && ch != ';')
            {
                is.setstate(std::ios_base::failbit);
                break;
            }
        } while (ch == ';');
        return is;
    }

    /**
     * Writes fcurve info to stream.
     * @relates fcurve
     */

    template <typename T, size_t N, typename Traits, typename CharT, typename CharTraits> std::basic_ostream<CharT, CharTraits> &
        operator << (std::basic_ostream<CharT, CharTraits> & os, const fcurve<T, N, Traits> & curve)
    {
        os << '[';
        switch (curve.type())
        {
        case fcurve_type::step:
            os << 's';
            break;
        case fcurve_type::linear:
            os << 'l';
            break;
        case fcurve_type::cubic:
            os << 'c';
            break;
        }
        
        for (size_t i = 0; i < curve.size(); i++)
        {
            os << ";";
            const typename fcurve<T, N, Traits>::key_type & key = curve.at(i);
            os << key.pos << ',' << key.dpos_left << ',' << key.dpos_right;
        }
        os << ']';
        return os;
    }

    /**
     * Commonly used instances for 2D, 3D, 4D and 5D cases.
     */
    
    template <typename T, typename Traits = float_traits<T> > using fcurve2 = fcurve<T, 2, Traits>;
    template <typename T, typename Traits = float_traits<T> > using fcurve3 = fcurve<T, 3, Traits>;
    template <typename T, typename Traits = float_traits<T> > using fcurve4 = fcurve<T, 4, Traits>;
    template <typename T, typename Traits = float_traits<T> > using fcurve5 = fcurve<T, 5, Traits>;
}

#endif
