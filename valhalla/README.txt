Hacked version of Valhalla routing engine
-----------------------------------------

Current version is based on Valhalla 3.0.9 release. Protobuf files are compiled using protoc 2.5.0.


Changed files
-------------

baldr/rapidjson_utils.h
baldr/graphtileheader.h
baldr/graphtile.h
baldr/graphreader.h
baldr/directededge.h
midgard/pointll.h
midgard/logging.h
sif/dynamiccost.h

worker.cc
baldr/verbal_text_formatter_factory.cc
baldr/transitroute.cc
baldr/streetnames_us.cc
baldr/streetnames.cc
baldr/streetnames_factory.cc
baldr/nodeinfo.cc
baldr/graphtileheader.cc
baldr/graphtile.cc
loki/worker.cc
meili/map_matcher.cc
odin/worker.cc
odin/narrative_builder_factory.cc
odin/maneuver.cc
odin/enhacedtrippath.cc
skadi/sample.cc
thor/worker.cc
thor/triplegbuilder.cc
thor/timedep_reverse.cc
thor/timedep_forward.cc
thor/route_matcher.cc
thor/map_matcher.cc
thor/costmatrix.cc

Most changes are marked with 'CARTOHACK' comments.
