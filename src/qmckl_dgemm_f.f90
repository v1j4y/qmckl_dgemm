module qmckl_dgemm
  use, intrinsic :: iso_c_binding
  integer  , parameter :: qmckl_packed_matrix = c_int64_t
  integer  , parameter :: qmckl_exit_code = c_int32_t
  integer(qmckl_exit_code), parameter :: QMCKL_SUCCESS                  = 0
  integer(qmckl_exit_code), parameter :: QMCKL_FAILURE                  = 101

  ! Fortran Interfaces

  interface
     integer (qmckl_packed_matrix) function qmckl_packed_matrix_create() bind(C)
       use, intrinsic :: iso_c_binding
       import
     end function qmckl_packed_matrix_create
  end interface

  interface
     integer (qmckl_exit_code) function qmckl_packed_matrix_destroy(packed_matrix) bind(C)
       use, intrinsic :: iso_c_binding
       import
       integer (qmckl_packed_matrix), intent(in), value :: packed_matrix
     end function qmckl_packed_matrix_destroy
  end interface

   interface
   integer(qmckl_exit_code) function qmckl_init_pack &
       (packed_matrix, mType, m, n, k) &
       bind(C)
     use, intrinsic :: iso_c_binding
     import
     implicit none

     integer (qmckl_packed_matrix) , intent(in)  , value :: packed_matrix
     character           , intent(in)  , value :: mType
     integer (c_int64_t) , intent(in)  , value :: m
     integer (c_int64_t) , intent(in)  , value :: n
     integer (c_int64_t) , intent(in)  , value :: k

   end function qmckl_init_pack
 end interface

   interface
   integer(qmckl_exit_code) function qmckl_pack_matrix &
       (packed_matrix, mType, m, n, A, lda) &
       bind(C)
     use, intrinsic :: iso_c_binding
     import
     implicit none

     integer (qmckl_packed_matrix) , intent(in)  , value :: packed_matrix
     character           , intent(in)  , value :: mType
     integer (c_int64_t) , intent(in)  , value :: m
     integer (c_int64_t) , intent(in)  , value :: n
     integer (c_int64_t) , intent(in)  , value :: lda
     real    (c_double ) , intent(in)          :: A(lda,*)
   end function qmckl_pack_matrix
 end interface

   interface
   integer(qmckl_exit_code) function qmckl_dgemm_tiled_avx2_nn &
       (A, lda, B, ldb, C, ldc) &
       bind(C)
     use, intrinsic :: iso_c_binding
     import
     implicit none

     integer (c_int64_t) , intent(in)  , value :: lda
     integer (qmckl_packed_matrix) , intent(in)  , value :: A
     integer (c_int64_t) , intent(in)  , value :: ldb
     integer (qmckl_packed_matrix) , intent(in)  , value :: B
     integer (c_int64_t) , intent(in)  , value :: ldc
     integer (qmckl_packed_matrix) , intent(in)  , value :: C

   end function qmckl_dgemm_tiled_avx2_nn
 end interface

   interface
   integer(qmckl_exit_code) function qmckl_dgemm_tiled_avx2 &
       (Min, Nin, Kin, A, lda, B, ldb, C, ldc) &
       bind(C, name='qmckl_dgemm_tiled')
     use, intrinsic :: iso_c_binding
     import
     implicit none

     integer (c_int64_t) , intent(in)  , value :: Min
     integer (c_int64_t) , intent(in)  , value :: Nin
     integer (c_int64_t) , intent(in)  , value :: Kin
     integer (c_int64_t) , intent(in)  , value :: lda
     real    (c_double ) , intent(in)          :: A(lda,*)
     integer (c_int64_t) , intent(in)  , value :: ldb
     real    (c_double ) , intent(in)          :: B(ldb,*)
     integer (c_int64_t) , intent(in)  , value :: ldc
     real    (c_double ) , intent(inout)       :: C(ldc,*)

   end function qmckl_dgemm_tiled_avx2
 end interface

   interface
   integer(qmckl_exit_code) function qmckl_unpack_matrix &
       (packed_matrix, A, m, n) &
       bind(C)
     use, intrinsic :: iso_c_binding
     import
     implicit none

     integer (qmckl_packed_matrix) , intent(in)  , value :: packed_matrix
     integer (c_int64_t) , intent(in)  , value :: m
     integer (c_int64_t) , intent(in)  , value :: n
     real    (c_double ) , intent(in)          :: A(n,*)

   end function qmckl_unpack_matrix
 end interface

end module qmckl_dgemm
