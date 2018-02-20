/*	For use with Bidi Reference Implementation
	For more information see the associated file bidi.cpp
 
	Credits:
	-------
	Written by: Asmus Freytag
	Command line interface by: Rick McGowan
	Verification (v24): Doug Felt
 
	Disclaimer and legal rights:
	---------------------------
	Copyright (C) 1999-2009, ASMUS, Inc. All Rights Reserved.
	Distributed under the Terms of Use in http://www.unicode.org/copyright.html.
 
	THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
	OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF THIRD PARTY RIGHTS.
	IN NO EVENT SHALL THE COPYRIGHT HOLDER OR HOLDERS INCLUDED IN THIS NOTICE
	BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL INDIRECT OR CONSEQUENTIAL DAMAGES,
	OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
	WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
	ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THE SOFTWARE.
 
 The files bidi.rc, and resource.h are distributed together with this file and are included
 in the above definition of software.
 */

#ifndef _BIDI_H_
#define _BIDI_H_

#include <stdint.h>

// For convenience of external callers, the following constitute the interface to the actual algorithm.
// For usage notes and paramter descriptions see the file bidi.cpp
#ifdef __cplusplus
extern "C" {
#endif

int bidi_run(uint32_t * pszLine, int * plevelLine, int cchLine, int * pfRTL);
int bidi_line(int baselevel, uint32_t * pszLine, int * pclsLine, int * plevelLine, int * plevelLineReorder, int cchPara, int fMirror, int * pbrk);
int bidi_paragraph(int *baselevel,  int * types, int * levels, int cch);

int bidi_classify(const uint32_t * pszText, int * pcls,int cch, int fWS);
int bidi_clean(uint32_t * pszInput, int cch);

#ifdef __cplusplus
}
#endif

#endif // _BIDI_H_
