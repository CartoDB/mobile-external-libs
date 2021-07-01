/*
* Block Ciphers
* (C) 2015 Jack Lloyd
*
* Botan is released under the Simplified BSD License (see license.txt)
*/

#include <botan/block_cipher.h>
#include <botan/scan_name.h>
#include <botan/exceptn.h>

#if defined(BOTAN_HAS_AES)
  #include <botan/aes.h>
#endif

#if defined(BOTAN_HAS_ARIA)
  #include <botan/aria.h>
#endif

#if defined(BOTAN_HAS_BLOWFISH)
  #include <botan/blowfish.h>
#endif

#if defined(BOTAN_HAS_CAMELLIA)
  #include <botan/camellia.h>
#endif

#if defined(BOTAN_HAS_CAST_128)
  #include <botan/cast128.h>
#endif

#if defined(BOTAN_HAS_CAST_256)
  #include <botan/cast256.h>
#endif

#if defined(BOTAN_HAS_CASCADE)
  #include <botan/cascade.h>
#endif

#if defined(BOTAN_HAS_DES)
  #include <botan/des.h>
  #include <botan/desx.h>
#endif

#if defined(BOTAN_HAS_GOST_28147_89)
  #include <botan/gost_28147.h>
#endif

#if defined(BOTAN_HAS_IDEA)
  #include <botan/idea.h>
#endif

#if defined(BOTAN_HAS_KASUMI)
  #include <botan/kasumi.h>
#endif

#if defined(BOTAN_HAS_LION)
  #include <botan/lion.h>
#endif

#if defined(BOTAN_HAS_MISTY1)
  #include <botan/misty1.h>
#endif

#if defined(BOTAN_HAS_NOEKEON)
  #include <botan/noekeon.h>
#endif

#if defined(BOTAN_HAS_SEED)
  #include <botan/seed.h>
#endif

#if defined(BOTAN_HAS_SERPENT)
  #include <botan/serpent.h>
#endif

#if defined(BOTAN_HAS_SHACAL2)
  #include <botan/shacal2.h>
#endif

#if defined(BOTAN_HAS_SM4)
  #include <botan/sm4.h>
#endif

#if defined(BOTAN_HAS_TWOFISH)
  #include <botan/twofish.h>
#endif

#if defined(BOTAN_HAS_THREEFISH_512)
  #include <botan/threefish_512.h>
#endif

#if defined(BOTAN_HAS_XTEA)
  #include <botan/xtea.h>
#endif

#if defined(BOTAN_HAS_OPENSSL)
  #include <botan/internal/openssl.h>
#endif

#if defined(BOTAN_HAS_COMMONCRYPTO)
  #include <botan/internal/commoncrypto.h>
#endif

namespace Botan {


/*
* RC5 Encryption
*/
void RC5::encrypt_n(const byte in[], byte out[], size_t blocks) const
   {
   for(size_t i = 0; i != blocks; ++i)
      {
      u32bit A = load_le<u32bit>(in, 0);
      u32bit B = load_le<u32bit>(in, 1);

      A += m_S[0]; B += m_S[1];
      for(size_t j = 0; j != m_rounds; j += 4)
         {
         A = rotate_left(A ^ B, B % 32) + m_S[2*j+2];
         B = rotate_left(B ^ A, A % 32) + m_S[2*j+3];

         A = rotate_left(A ^ B, B % 32) + m_S[2*j+4];
         B = rotate_left(B ^ A, A % 32) + m_S[2*j+5];

         A = rotate_left(A ^ B, B % 32) + m_S[2*j+6];
         B = rotate_left(B ^ A, A % 32) + m_S[2*j+7];

         A = rotate_left(A ^ B, B % 32) + m_S[2*j+8];
         B = rotate_left(B ^ A, A % 32) + m_S[2*j+9];
         }

      store_le(out, A, B);

      in += BLOCK_SIZE;
      out += BLOCK_SIZE;
      }
   }

/*
* RC5 Decryption
*/
void RC5::decrypt_n(const byte in[], byte out[], size_t blocks) const
   {
   for(size_t i = 0; i != blocks; ++i)
      {
      u32bit A = load_le<u32bit>(in, 0);
      u32bit B = load_le<u32bit>(in, 1);

      for(size_t j = m_rounds; j != 0; j -= 4)
         {
         B = rotate_right(B - m_S[2*j+1], A % 32) ^ A;
         A = rotate_right(A - m_S[2*j  ], B % 32) ^ B;

         B = rotate_right(B - m_S[2*j-1], A % 32) ^ A;
         A = rotate_right(A - m_S[2*j-2], B % 32) ^ B;

         B = rotate_right(B - m_S[2*j-3], A % 32) ^ A;
         A = rotate_right(A - m_S[2*j-4], B % 32) ^ B;

         B = rotate_right(B - m_S[2*j-5], A % 32) ^ A;
         A = rotate_right(A - m_S[2*j-6], B % 32) ^ B;
         }
      B -= m_S[1]; A -= m_S[0];

      store_le(out, A, B);

      in += BLOCK_SIZE;
      out += BLOCK_SIZE;
      }
   }

/*
* RC5 Key Schedule
*/
void RC5::key_schedule(const byte key[], size_t length)
   {
   m_S.resize(2*m_rounds + 2);

   const size_t WORD_KEYLENGTH = (((length - 1) / 4) + 1);
   const size_t MIX_ROUNDS     = 3 * std::max(WORD_KEYLENGTH, m_S.size());

   m_S[0] = 0xB7E15163;
   for(size_t i = 1; i != m_S.size(); ++i)
      m_S[i] = m_S[i-1] + 0x9E3779B9;

   secure_vector<u32bit> K(8);

   for(s32bit i = length-1; i >= 0; --i)
      K[i/4] = (K[i/4] << 8) + key[i];

   u32bit A = 0, B = 0;

   for(size_t i = 0; i != MIX_ROUNDS; ++i)
      {
      A = rotate_left(m_S[i % m_S.size()] + A + B, 3);
      B = rotate_left(K[i % WORD_KEYLENGTH] + A + B, (A + B) % 32);
      m_S[i % m_S.size()] = A;
      K[i % WORD_KEYLENGTH] = B;
      }
   }

void RC5::clear()
   {
   zap(m_S);
   }

/*
* Return the name of this type
*/
std::string RC5::name() const
   {
   return "RC5(" + std::to_string(m_rounds) + ")";
   }

/*
* RC5 Constructor
*/
RC5::RC5(size_t r) : m_rounds(r)
   {
   if(m_rounds < 8 || m_rounds > 32 || (m_rounds % 4 != 0))
      throw Invalid_Argument("RC5: Invalid number of rounds " +
                             std::to_string(m_rounds));
   }

}


namespace Botan {

std::unique_ptr<BlockCipher>
BlockCipher::create(const std::string& algo,
                    const std::string& provider)
   {
   if(algo == "RC5")
      {
      return std::unique_ptr<BlockCipher>(new RC5(16));
      }

#if defined(BOTAN_HAS_COMMONCRYPTO)
   if(provider.empty() || provider == "commoncrypto")
      {
      if(auto bc = make_commoncrypto_block_cipher(algo))
         return bc;

      if(!provider.empty())
         return nullptr;
      }
#endif

#if defined(BOTAN_HAS_OPENSSL)
   if(provider.empty() || provider == "openssl")
      {
      if(auto bc = make_openssl_block_cipher(algo))
         return bc;

      if(!provider.empty())
         return nullptr;
      }
#endif

   // TODO: CryptoAPI
   // TODO: /dev/crypto

   // Only base providers from here on out
   if(provider.empty() == false && provider != "base")
      return nullptr;

#if defined(BOTAN_HAS_AES)
   if(algo == "AES-128")
      {
      return std::unique_ptr<BlockCipher>(new AES_128);
      }

   if(algo == "AES-192")
      {
      return std::unique_ptr<BlockCipher>(new AES_192);
      }

   if(algo == "AES-256")
      {
      return std::unique_ptr<BlockCipher>(new AES_256);
      }
#endif

#if defined(BOTAN_HAS_ARIA)
   if(algo == "ARIA-128")
      {
      return std::unique_ptr<BlockCipher>(new ARIA_128);
      }

   if(algo == "ARIA-192")
      {
      return std::unique_ptr<BlockCipher>(new ARIA_192);
      }

   if(algo == "ARIA-256")
      {
      return std::unique_ptr<BlockCipher>(new ARIA_256);
      }
#endif

#if defined(BOTAN_HAS_SERPENT)
   if(algo == "Serpent")
      {
      return std::unique_ptr<BlockCipher>(new Serpent);
      }
#endif

#if defined(BOTAN_HAS_SHACAL2)
   if(algo == "SHACAL2")
      {
      return std::unique_ptr<BlockCipher>(new SHACAL2);
      }
#endif

#if defined(BOTAN_HAS_TWOFISH)
   if(algo == "Twofish")
      {
      return std::unique_ptr<BlockCipher>(new Twofish);
      }
#endif

#if defined(BOTAN_HAS_THREEFISH_512)
   if(algo == "Threefish-512")
      {
      return std::unique_ptr<BlockCipher>(new Threefish_512);
      }
#endif

#if defined(BOTAN_HAS_BLOWFISH)
   if(algo == "Blowfish")
      {
      return std::unique_ptr<BlockCipher>(new Blowfish);
      }
#endif

#if defined(BOTAN_HAS_CAMELLIA)
   if(algo == "Camellia-128")
      {
      return std::unique_ptr<BlockCipher>(new Camellia_128);
      }

   if(algo == "Camellia-192")
      {
      return std::unique_ptr<BlockCipher>(new Camellia_192);
      }

   if(algo == "Camellia-256")
      {
      return std::unique_ptr<BlockCipher>(new Camellia_256);
      }
#endif

#if defined(BOTAN_HAS_DES)
   if(algo == "DES")
      {
      return std::unique_ptr<BlockCipher>(new DES);
      }

   if(algo == "DESX")
      {
      return std::unique_ptr<BlockCipher>(new DESX);
      }

   if(algo == "TripleDES" || algo == "3DES" || algo == "DES-EDE")
      {
      return std::unique_ptr<BlockCipher>(new TripleDES);
      }
#endif

#if defined(BOTAN_HAS_NOEKEON)
   if(algo == "Noekeon")
      {
      return std::unique_ptr<BlockCipher>(new Noekeon);
      }
#endif

#if defined(BOTAN_HAS_CAST_128)
   if(algo == "CAST-128" || algo == "CAST5")
      {
      return std::unique_ptr<BlockCipher>(new CAST_128);
      }
#endif

#if defined(BOTAN_HAS_CAST_256)
   if(algo == "CAST-256")
      {
      return std::unique_ptr<BlockCipher>(new CAST_256);
      }
#endif

#if defined(BOTAN_HAS_IDEA)
   if(algo == "IDEA")
      {
      return std::unique_ptr<BlockCipher>(new IDEA);
      }
#endif

#if defined(BOTAN_HAS_KASUMI)
   if(algo == "KASUMI")
      {
      return std::unique_ptr<BlockCipher>(new KASUMI);
      }
#endif

#if defined(BOTAN_HAS_MISTY1)
   if(algo == "MISTY1")
      {
      return std::unique_ptr<BlockCipher>(new MISTY1);
      }
#endif

#if defined(BOTAN_HAS_SEED)
   if(algo == "SEED")
      {
      return std::unique_ptr<BlockCipher>(new SEED);
      }
#endif

#if defined(BOTAN_HAS_SM4)
   if(algo == "SM4")
      {
      return std::unique_ptr<BlockCipher>(new SM4);
      }
#endif

#if defined(BOTAN_HAS_XTEA)
   if(algo == "XTEA")
      {
      return std::unique_ptr<BlockCipher>(new XTEA);
      }
#endif

   const SCAN_Name req(algo);

#if defined(BOTAN_HAS_GOST_28147_89)
   if(req.algo_name() == "GOST-28147-89")
      {
      return std::unique_ptr<BlockCipher>(new GOST_28147_89(req.arg(0, "R3411_94_TestParam")));
      }
#endif

#if defined(BOTAN_HAS_CASCADE)
   if(req.algo_name() == "Cascade" && req.arg_count() == 2)
      {
      std::unique_ptr<BlockCipher> c1(BlockCipher::create(req.arg(0)));
      std::unique_ptr<BlockCipher> c2(BlockCipher::create(req.arg(1)));

      if(c1 && c2)
         return std::unique_ptr<BlockCipher>(new Cascade_Cipher(c1.release(), c2.release()));
      }
#endif

#if defined(BOTAN_HAS_LION)
   if(req.algo_name() == "Lion" && req.arg_count_between(2, 3))
      {
      std::unique_ptr<HashFunction> hash(HashFunction::create(req.arg(0)));
      std::unique_ptr<StreamCipher> stream(StreamCipher::create(req.arg(1)));

      if(hash && stream)
         {
         const size_t block_size = req.arg_as_integer(2, 1024);
         return std::unique_ptr<BlockCipher>(new Lion(hash.release(), stream.release(), block_size));
         }
      }
#endif

   BOTAN_UNUSED(req);
   BOTAN_UNUSED(provider);

   return nullptr;
   }

//static
std::unique_ptr<BlockCipher>
BlockCipher::create_or_throw(const std::string& algo,
                             const std::string& provider)
   {
   if(auto bc = BlockCipher::create(algo, provider))
      {
      return bc;
      }
   throw Lookup_Error("Block cipher", algo, provider);
   }

std::vector<std::string> BlockCipher::providers(const std::string& algo)
   {
   return probe_providers_of<BlockCipher>(algo, { "base", "openssl", "commoncrypto" });
   }

}
