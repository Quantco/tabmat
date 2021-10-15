#pragma once

#ifndef _WIN32
  #define JEMALLOC_NO_DEMANGLE
  #if __APPLE__
    #if not defined(JEMALLOC_INSTALL_SUFFIX)
      #define JEMALLOC_NO_RENAME
    #endif
  #endif
  // Compute jemalloc include path
  #define STRINGIFY(X) STRINGIFY2(X)
  #define STRINGIFY2(X) #X
  #define CAT(X,Y) CAT2(X,Y)
  #define CAT2(X,Y) X##Y
  #define JE_INCLUDE STRINGIFY(CAT(jemalloc/jemalloc,JEMALLOC_INSTALL_SUFFIX).h)
  #include JE_INCLUDE
#endif

#include <functional>

std::size_t round_to_align(std::size_t size, std::size_t alignment) {
  std::size_t remainder = size % alignment;

  if (remainder == 0) {
    return size;
  } else {
    return size + alignment - remainder;
  }
}

// This function returns a safe smart pointer that will properly delete a
// segment of alligned memory.
// Internally, it handles distinguishing between Windows and non-Windows
// allocation.
template <typename F>
std::unique_ptr<F, std::function<void(F*)>> make_aligned_unique(std::size_t size, std::size_t alignment) {
    std::size_t aligned_size = round_to_align(size * sizeof(F), alignment);

#ifndef _WIN32
    F* out = static_cast<F*>(je_aligned_alloc(alignment, aligned_size));
#else
    F* out = static_cast<F*>(_aligned_malloc(aligned_size, alignment));
#endif
    return std::unique_ptr<F, std::function<void(F*)>>(
        out, 
        [=] (F* ptr) { 
#ifndef _WIN32
            je_sdallocx(ptr, aligned_size, 0);
#else
            _aligned_free(ptr);
#endif
        }
    );
}
