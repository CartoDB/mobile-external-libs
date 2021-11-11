Hacked version of Botan crypto library
-----------------------------------------

Current version is based on Botan 2.8.2 release.

Changed files
-------------

botan/src/lib/block/block_cipher.h
botan/src/lib/block/block_cipher.cpp

These changes bring back RC5 block cipher.


How to generate
---------------

python configure.py --compiler generic --os none --amalgamation --cc clang --cpu generic --minimized-build --enable-modules=sha1,md5,hex,base64,dsa,emsa1,asn1,block,pubkey,cbc
mv botan_all.cpp botan_all_clang.cpp
python configure.py --compiler generic --os none --amalgamation --cc msvc --cpu generic --minimized-build --enable-modules=sha1,md5,hex,base64,dsa,emsa1,asn1,block,pubkey,cbc
mv botan_all.cpp botan_all_msvc.cpp

Remove BOTAN_BUILD_COMPILER_IS_MSVC and BOTAN_BUILD_COMPILER_IS_CLANG from botan_all.h
