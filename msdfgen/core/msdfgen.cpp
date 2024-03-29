
#include "../msdfgen.h"

#include "arithmetics.hpp"

#include <cmath>

namespace msdfgen {

struct MultiDistance {
    double r, g, b;
    double med;
};

struct EdgeBounds {
    double l, b, r, t;
};

struct EdgeState {
    EdgeBounds bounds;
    Point2 closestPoint;
    double closestDist;
};

static inline bool pixelClash(const FloatRGB &a, const FloatRGB &b, double threshold) {
    // Only consider pair where both are on the inside or both are on the outside
    bool aIn = (a.r > .5f)+(a.g > .5f)+(a.b > .5f) >= 2;
    bool bIn = (b.r > .5f)+(b.g > .5f)+(b.b > .5f) >= 2;
    if (aIn != bIn) return false;
    // If the change is 0 <-> 1 or 2 <-> 3 channels and not 1 <-> 1 or 2 <-> 2, it is not a clash
    if ((a.r > .5f && a.g > .5f && a.b > .5f) || (a.r < .5f && a.g < .5f && a.b < .5f)
        || (b.r > .5f && b.g > .5f && b.b > .5f) || (b.r < .5f && b.g < .5f && b.b < .5f))
        return false;
    // Find which color is which: _a, _b = the changing channels, _c = the remaining one
    float aa, ab, ba, bb, ac, bc;
    if ((a.r > .5f) != (b.r > .5f) && (a.r < .5f) != (b.r < .5f)) {
        aa = a.r, ba = b.r;
        if ((a.g > .5f) != (b.g > .5f) && (a.g < .5f) != (b.g < .5f)) {
            ab = a.g, bb = b.g;
            ac = a.b, bc = b.b;
        } else if ((a.b > .5f) != (b.b > .5f) && (a.b < .5f) != (b.b < .5f)) {
            ab = a.b, bb = b.b;
            ac = a.g, bc = b.g;
        } else
            return false; // this should never happen
    } else if ((a.g > .5f) != (b.g > .5f) && (a.g < .5f) != (b.g < .5f)
        && (a.b > .5f) != (b.b > .5f) && (a.b < .5f) != (b.b < .5f)) {
        aa = a.g, ba = b.g;
        ab = a.b, bb = b.b;
        ac = a.r, bc = b.r;
    } else
        return false;
    // Find if the channels are in fact discontinuous
    return (fabsf(aa-ba) >= threshold)
        && (fabsf(ab-bb) >= threshold)
        && fabsf(ac-.5f) >= fabsf(bc-.5f); // Out of the pair, only flag the pixel farther from a shape edge
}

static std::vector<EdgeState> buildEdgeState(const Shape &shape) {
    std::vector<EdgeState> edgeState;
    for (std::vector<Contour>::const_iterator contour = shape.contours.begin(); contour != shape.contours.end(); ++contour) {
        for (std::vector<EdgeHolder>::const_iterator edge = contour->edges.begin(); edge != contour->edges.end(); ++edge) {
            EdgeBounds bounds;
            bounds.l = bounds.b = fabs(SignedDistance::INFINITE.distance);
            bounds.r = bounds.t = -fabs(SignedDistance::INFINITE.distance);
            (*edge)->bounds(bounds.l, bounds.b, bounds.r, bounds.t);
            EdgeState state;
            state.bounds = bounds;
            state.closestPoint = Point2(0, 0);
            state.closestDist = -1;
            edgeState.push_back(state);
        }
    }
    return edgeState;
}

void msdfErrorCorrection(Bitmap<FloatRGB> &output, const Vector2 &threshold) {
    std::vector<std::pair<int, int> > clashes;
    int w = output.width(), h = output.height();
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            if ((x > 0 && pixelClash(output(x, y), output(x-1, y), threshold.x))
                || (x < w-1 && pixelClash(output(x, y), output(x+1, y), threshold.x))
                || (y > 0 && pixelClash(output(x, y), output(x, y-1), threshold.y))
                || (y < h-1 && pixelClash(output(x, y), output(x, y+1), threshold.y)))
                clashes.push_back(std::make_pair(x, y));
        }
    for (std::vector<std::pair<int, int> >::const_iterator clash = clashes.begin(); clash != clashes.end(); ++clash) {
        FloatRGB &pixel = output(clash->first, clash->second);
        float med = median(pixel.r, pixel.g, pixel.b);
        pixel.r = med, pixel.g = med, pixel.b = med;
    }
}

void generateSDF(Bitmap<float> &output, const Shape &shape, double range, const Vector2 &scale, const Vector2 &translate, double maxValue) {
    int contourCount = shape.contours.size();
    int w = output.width(), h = output.height();
    std::vector<int> windings;
    windings.reserve(contourCount);
    for (std::vector<Contour>::const_iterator contour = shape.contours.begin(); contour != shape.contours.end(); ++contour)
        windings.push_back(contour->winding());
    std::vector<EdgeState> edgeState = buildEdgeState(shape);

#ifdef MSDFGEN_USE_OPENMP
    #pragma omp parallel
#endif
    {
        std::vector<double> contourSD;
        contourSD.resize(contourCount);
#ifdef MSDFGEN_USE_OPENMP
        #pragma omp for
#endif
        EdgeHolder closestEdge;
        for (int y = 0; y < h; ++y) {
            int row = shape.inverseYAxis ? h-y-1 : y;
            for (int x = 0; x < w; ++x) {
                double dummy;
                Point2 p = Vector2(x+.5, y+.5)/scale-translate;
                double negDist = -SignedDistance::INFINITE.distance;
                double posDist = SignedDistance::INFINITE.distance;
                int winding = 0;

                std::vector<Contour>::const_iterator contour = shape.contours.begin();
                std::vector<EdgeState>::const_iterator state = edgeState.begin();
                for (int i = 0; i < contourCount; ++i, ++contour) {
                    SignedDistance minDistance(-maxValue, 1);
                    if (closestEdge) {
                        SignedDistance distance = closestEdge->signedDistance(p, dummy);
                        if (distance < minDistance)
                            minDistance = distance;
                    }

                    for (std::vector<EdgeHolder>::const_iterator edge = contour->edges.begin(); edge != contour->edges.end(); ++edge, ++state) {
                        double absDist = fabs(minDistance.distance);
                        if (p.x + absDist < state->bounds.l || state->bounds.r + absDist < p.x || p.y + absDist < state->bounds.b || state->bounds.t + absDist < p.y)
                            continue;

                        SignedDistance distance = (*edge)->signedDistance(p, dummy);
                        if (distance < minDistance) {
                            minDistance = distance;
                            closestEdge = *edge;
                        }
                    }
                    contourSD[i] = minDistance.distance;
                    if (windings[i] > 0 && minDistance.distance >= 0 && fabs(minDistance.distance) < fabs(posDist))
                        posDist = minDistance.distance;
                    if (windings[i] < 0 && minDistance.distance <= 0 && fabs(minDistance.distance) < fabs(negDist))
                        negDist = minDistance.distance;
                }

                double sd = SignedDistance::INFINITE.distance;
                if (posDist >= 0 && fabs(posDist) <= fabs(negDist)) {
                    sd = posDist;
                    winding = 1;
                    for (int i = 0; i < contourCount; ++i)
                        if (windings[i] > 0 && contourSD[i] > sd && fabs(contourSD[i]) < fabs(negDist))
                            sd = contourSD[i];
                } else if (negDist <= 0 && fabs(negDist) <= fabs(posDist)) {
                    sd = negDist;
                    winding = -1;
                    for (int i = 0; i < contourCount; ++i)
                        if (windings[i] < 0 && contourSD[i] < sd && fabs(contourSD[i]) < fabs(posDist))
                            sd = contourSD[i];
                }
                for (int i = 0; i < contourCount; ++i)
                    if (windings[i] != winding && fabs(contourSD[i]) < fabs(sd))
                        sd = contourSD[i];

                output(x, row) = float(sd/range+.5);
            }
        }
    }
}

void generatePseudoSDF(Bitmap<float> &output, const Shape &shape, double range, const Vector2 &scale, const Vector2 &translate) {
    int contourCount = shape.contours.size();
    int w = output.width(), h = output.height();
    std::vector<int> windings;
    windings.reserve(contourCount);
    for (std::vector<Contour>::const_iterator contour = shape.contours.begin(); contour != shape.contours.end(); ++contour)
        windings.push_back(contour->winding());

#ifdef MSDFGEN_USE_OPENMP
    #pragma omp parallel
#endif
    {
        std::vector<double> contourSD;
        contourSD.resize(contourCount);
#ifdef MSDFGEN_USE_OPENMP
        #pragma omp for
#endif
        for (int y = 0; y < h; ++y) {
            int row = shape.inverseYAxis ? h-y-1 : y;
            for (int x = 0; x < w; ++x) {
                Point2 p = Vector2(x+.5, y+.5)/scale-translate;
                double sd = SignedDistance::INFINITE.distance;
                double negDist = -SignedDistance::INFINITE.distance;
                double posDist = SignedDistance::INFINITE.distance;
                int winding = 0;

                std::vector<Contour>::const_iterator contour = shape.contours.begin();
                for (int i = 0; i < contourCount; ++i, ++contour) {
                    SignedDistance minDistance;
                    const EdgeHolder *nearEdge = NULL;
                    double nearParam = 0;
                    for (std::vector<EdgeHolder>::const_iterator edge = contour->edges.begin(); edge != contour->edges.end(); ++edge) {
                        double param;
                        SignedDistance distance = (*edge)->signedDistance(p, param);
                        if (distance < minDistance) {
                            minDistance = distance;
                            nearEdge = &*edge;
                            nearParam = param;
                        }
                    }
                    if (fabs(minDistance.distance) < fabs(sd)) {
                        sd = minDistance.distance;
                        winding = -windings[i];
                    }
                    if (nearEdge)
                        (*nearEdge)->distanceToPseudoDistance(minDistance, p, nearParam);
                    contourSD[i] = minDistance.distance;
                    if (windings[i] > 0 && minDistance.distance >= 0 && fabs(minDistance.distance) < fabs(posDist))
                        posDist = minDistance.distance;
                    if (windings[i] < 0 && minDistance.distance <= 0 && fabs(minDistance.distance) < fabs(negDist))
                        negDist = minDistance.distance;
                }

                double psd = SignedDistance::INFINITE.distance;
                if (posDist >= 0 && fabs(posDist) <= fabs(negDist)) {
                    psd = posDist;
                    winding = 1;
                    for (int i = 0; i < contourCount; ++i)
                        if (windings[i] > 0 && contourSD[i] > psd && fabs(contourSD[i]) < fabs(negDist))
                            psd = contourSD[i];
                } else if (negDist <= 0 && fabs(negDist) <= fabs(posDist)) {
                    psd = negDist;
                    winding = -1;
                    for (int i = 0; i < contourCount; ++i)
                        if (windings[i] < 0 && contourSD[i] < psd && fabs(contourSD[i]) < fabs(posDist))
                            psd = contourSD[i];
                }
                for (int i = 0; i < contourCount; ++i)
                    if (windings[i] != winding && fabs(contourSD[i]) < fabs(psd))
                        psd = contourSD[i];

                output(x, row) = float(psd/range+.5);
            }
        }
    }
}

void generateMSDF(Bitmap<FloatRGB> &output, const Shape &shape, double range, const Vector2 &scale, const Vector2 &translate, double edgeThreshold) {
    int contourCount = shape.contours.size();
    int w = output.width(), h = output.height();
    std::vector<int> windings;
    windings.reserve(contourCount);
    for (std::vector<Contour>::const_iterator contour = shape.contours.begin(); contour != shape.contours.end(); ++contour)
        windings.push_back(contour->winding());

#ifdef MSDFGEN_USE_OPENMP
    #pragma omp parallel
#endif
    {
        std::vector<MultiDistance> contourSD;
        contourSD.resize(contourCount);
#ifdef MSDFGEN_USE_OPENMP
        #pragma omp for
#endif
        for (int y = 0; y < h; ++y) {
            int row = shape.inverseYAxis ? h-y-1 : y;
            for (int x = 0; x < w; ++x) {
                Point2 p = Vector2(x+.5, y+.5)/scale-translate;

                struct EdgePoint {
                    SignedDistance minDistance;
                    const EdgeHolder *nearEdge;
                    double nearParam;
                } sr, sg, sb;
                sr.nearEdge = sg.nearEdge = sb.nearEdge = NULL;
                sr.nearParam = sg.nearParam = sb.nearParam = 0;
                double d = fabs(SignedDistance::INFINITE.distance);
                double negDist = -SignedDistance::INFINITE.distance;
                double posDist = SignedDistance::INFINITE.distance;
                int winding = 0;

                std::vector<Contour>::const_iterator contour = shape.contours.begin();
                for (int i = 0; i < contourCount; ++i, ++contour) {
                    EdgePoint r, g, b;
                    r.nearEdge = g.nearEdge = b.nearEdge = NULL;
                    r.nearParam = g.nearParam = b.nearParam = 0;

                    for (std::vector<EdgeHolder>::const_iterator edge = contour->edges.begin(); edge != contour->edges.end(); ++edge) {
                        double param;
                        SignedDistance distance = (*edge)->signedDistance(p, param);
                        if ((*edge)->color&RED && distance < r.minDistance) {
                            r.minDistance = distance;
                            r.nearEdge = &*edge;
                            r.nearParam = param;
                        }
                        if ((*edge)->color&GREEN && distance < g.minDistance) {
                            g.minDistance = distance;
                            g.nearEdge = &*edge;
                            g.nearParam = param;
                        }
                        if ((*edge)->color&BLUE && distance < b.minDistance) {
                            b.minDistance = distance;
                            b.nearEdge = &*edge;
                            b.nearParam = param;
                        }
                    }
                    if (r.minDistance < sr.minDistance)
                        sr = r;
                    if (g.minDistance < sg.minDistance)
                        sg = g;
                    if (b.minDistance < sb.minDistance)
                        sb = b;

                    double medMinDistance = fabs(median(r.minDistance.distance, g.minDistance.distance, b.minDistance.distance));
                    if (medMinDistance < d) {
                        d = medMinDistance;
                        winding = -windings[i];
                    }
                    if (r.nearEdge)
                        (*r.nearEdge)->distanceToPseudoDistance(r.minDistance, p, r.nearParam);
                    if (g.nearEdge)
                        (*g.nearEdge)->distanceToPseudoDistance(g.minDistance, p, g.nearParam);
                    if (b.nearEdge)
                        (*b.nearEdge)->distanceToPseudoDistance(b.minDistance, p, b.nearParam);
                    medMinDistance = median(r.minDistance.distance, g.minDistance.distance, b.minDistance.distance);
                    contourSD[i].r = r.minDistance.distance;
                    contourSD[i].g = g.minDistance.distance;
                    contourSD[i].b = b.minDistance.distance;
                    contourSD[i].med = medMinDistance;
                    if (windings[i] > 0 && medMinDistance >= 0 && fabs(medMinDistance) < fabs(posDist))
                        posDist = medMinDistance;
                    if (windings[i] < 0 && medMinDistance <= 0 && fabs(medMinDistance) < fabs(negDist))
                        negDist = medMinDistance;
                }
                if (sr.nearEdge)
                    (*sr.nearEdge)->distanceToPseudoDistance(sr.minDistance, p, sr.nearParam);
                if (sg.nearEdge)
                    (*sg.nearEdge)->distanceToPseudoDistance(sg.minDistance, p, sg.nearParam);
                if (sb.nearEdge)
                    (*sb.nearEdge)->distanceToPseudoDistance(sb.minDistance, p, sb.nearParam);

                MultiDistance msd;
                msd.r = msd.g = msd.b = msd.med = SignedDistance::INFINITE.distance;
                if (posDist >= 0 && fabs(posDist) <= fabs(negDist)) {
                    msd.med = SignedDistance::INFINITE.distance;
                    winding = 1;
                    for (int i = 0; i < contourCount; ++i)
                        if (windings[i] > 0 && contourSD[i].med > msd.med && fabs(contourSD[i].med) < fabs(negDist))
                            msd = contourSD[i];
                } else if (negDist <= 0 && fabs(negDist) <= fabs(posDist)) {
                    msd.med = -SignedDistance::INFINITE.distance;
                    winding = -1;
                    for (int i = 0; i < contourCount; ++i)
                        if (windings[i] < 0 && contourSD[i].med < msd.med && fabs(contourSD[i].med) < fabs(posDist))
                            msd = contourSD[i];
                }
                for (int i = 0; i < contourCount; ++i)
                    if (windings[i] != winding && fabs(contourSD[i].med) < fabs(msd.med))
                        msd = contourSD[i];
                if (median(sr.minDistance.distance, sg.minDistance.distance, sb.minDistance.distance) == msd.med) {
                    msd.r = sr.minDistance.distance;
                    msd.g = sg.minDistance.distance;
                    msd.b = sb.minDistance.distance;
                }

                output(x, row).r = float(msd.r/range+.5);
                output(x, row).g = float(msd.g/range+.5);
                output(x, row).b = float(msd.b/range+.5);
            }
        }
    }

    if (edgeThreshold > 0)
        msdfErrorCorrection(output, edgeThreshold/(scale*range));
}

void generateSDF_legacy(Bitmap<float> &output, const Shape &shape, double range, const Vector2 &scale, const Vector2 &translate, double maxValue) {
    int w = output.width(), h = output.height();
    std::vector<EdgeState> edgeState = buildEdgeState(shape);
#ifdef MSDFGEN_USE_OPENMP
    #pragma omp parallel for
#endif
    EdgeHolder closestEdge;
    std::vector<EdgeState>::iterator closestState;
    for (int y = 0; y < h; ++y) {
        int row = shape.inverseYAxis ? h-y-1 : y;
        for (int n = 0; n < w; n++) {
            int x = (y & 1 ? w - n - 1 : n);
            double dummy;
            Point2 p = Vector2(x + .5, y + .5) / scale - translate;
            SignedDistance minDistance(-maxValue, 1);
            if (closestEdge) {
                double absDist = std::abs(minDistance.distance);
                float dx = static_cast<float>(closestState->closestPoint.x - p.x), dy = static_cast<float>(closestState->closestPoint.y - p.y);
                if (absDist + std::sqrt(dx * dx + dy * dy) > closestState->closestDist) {
                    SignedDistance distance = closestEdge->signedDistance(p, dummy);
                    closestState->closestPoint = p;
                    closestState->closestDist = std::abs(distance.distance);
                    if (distance < minDistance) {
                        minDistance = distance;
                    }
                }
            }
            
            std::vector<EdgeState>::iterator state = edgeState.begin();
            for (std::vector<Contour>::const_iterator contour = shape.contours.begin(); contour != shape.contours.end(); ++contour) {
                for (std::vector<EdgeHolder>::const_iterator edge = contour->edges.begin(); edge != contour->edges.end(); ++edge, ++state) {
                    double absDist = std::abs(minDistance.distance);
                    if (p.x + absDist < state->bounds.l || state->bounds.r + absDist < p.x || p.y + absDist < state->bounds.b || state->bounds.t + absDist < p.y)
                        continue;
                    
                    float dx = static_cast<float>(state->closestPoint.x - p.x), dy = static_cast<float>(state->closestPoint.y - p.y);
                    if (absDist + std::sqrt(dx * dx + dy * dy) > state->closestDist) {
                        SignedDistance distance = (*edge)->signedDistance(p, dummy);
                        state->closestPoint = p;
                        state->closestDist = std::abs(distance.distance);
                        if (distance < minDistance) {
                            minDistance = distance;
                            closestEdge = *edge;
                            closestState = state;
                        }
                    }
                }
            }
            
            output(x, row) = float(minDistance.distance / range + .5);
        }
    }
}

void generatePseudoSDF_legacy(Bitmap<float> &output, const Shape &shape, double range, const Vector2 &scale, const Vector2 &translate) {
    int w = output.width(), h = output.height();
#ifdef MSDFGEN_USE_OPENMP
    #pragma omp parallel for
#endif
    for (int y = 0; y < h; ++y) {
        int row = shape.inverseYAxis ? h-y-1 : y;
        for (int x = 0; x < w; ++x) {
            Point2 p = Vector2(x+.5, y+.5)/scale-translate;
            SignedDistance minDistance;
            const EdgeHolder *nearEdge = NULL;
            double nearParam = 0;
            for (std::vector<Contour>::const_iterator contour = shape.contours.begin(); contour != shape.contours.end(); ++contour)
                for (std::vector<EdgeHolder>::const_iterator edge = contour->edges.begin(); edge != contour->edges.end(); ++edge) {
                    double param;
                    SignedDistance distance = (*edge)->signedDistance(p, param);
                    if (distance < minDistance) {
                        minDistance = distance;
                        nearEdge = &*edge;
                        nearParam = param;
                    }
                }
            if (nearEdge)
                (*nearEdge)->distanceToPseudoDistance(minDistance, p, nearParam);
            output(x, row) = float(minDistance.distance/range+.5);
        }
    }
}

void generateMSDF_legacy(Bitmap<FloatRGB> &output, const Shape &shape, double range, const Vector2 &scale, const Vector2 &translate, double edgeThreshold) {
    int w = output.width(), h = output.height();
#ifdef MSDFGEN_USE_OPENMP
    #pragma omp parallel for
#endif
    for (int y = 0; y < h; ++y) {
        int row = shape.inverseYAxis ? h-y-1 : y;
        for (int x = 0; x < w; ++x) {
            Point2 p = Vector2(x+.5, y+.5)/scale-translate;

            struct {
                SignedDistance minDistance;
                const EdgeHolder *nearEdge;
                double nearParam;
            } r, g, b;
            r.nearEdge = g.nearEdge = b.nearEdge = NULL;
            r.nearParam = g.nearParam = b.nearParam = 0;

            for (std::vector<Contour>::const_iterator contour = shape.contours.begin(); contour != shape.contours.end(); ++contour)
                for (std::vector<EdgeHolder>::const_iterator edge = contour->edges.begin(); edge != contour->edges.end(); ++edge) {
                    double param;
                    SignedDistance distance = (*edge)->signedDistance(p, param);
                    if ((*edge)->color&RED && distance < r.minDistance) {
                        r.minDistance = distance;
                        r.nearEdge = &*edge;
                        r.nearParam = param;
                    }
                    if ((*edge)->color&GREEN && distance < g.minDistance) {
                        g.minDistance = distance;
                        g.nearEdge = &*edge;
                        g.nearParam = param;
                    }
                    if ((*edge)->color&BLUE && distance < b.minDistance) {
                        b.minDistance = distance;
                        b.nearEdge = &*edge;
                        b.nearParam = param;
                    }
                }

            if (r.nearEdge)
                (*r.nearEdge)->distanceToPseudoDistance(r.minDistance, p, r.nearParam);
            if (g.nearEdge)
                (*g.nearEdge)->distanceToPseudoDistance(g.minDistance, p, g.nearParam);
            if (b.nearEdge)
                (*b.nearEdge)->distanceToPseudoDistance(b.minDistance, p, b.nearParam);
            output(x, row).r = float(r.minDistance.distance/range+.5);
            output(x, row).g = float(g.minDistance.distance/range+.5);
            output(x, row).b = float(b.minDistance.distance/range+.5);
        }
    }

    if (edgeThreshold > 0)
        msdfErrorCorrection(output, edgeThreshold/(scale*range));
}

}
