module qmckl_dgemm
  use, intrinsic :: iso_c_binding
  integer  , parameter :: qmckl_context = c_int64_t
  integer*8, parameter :: QMCKL_NULL_CONTEXT = 0 
  integer  , parameter :: qmckl_exit_code = c_int32_t
  integer(qmckl_exit_code), parameter :: QMCKL_SUCCESS                  = 0
  integer(qmckl_exit_code), parameter :: QMCKL_FAILURE                  = 101

  ! Fortran Interfaces
  interface
     integer (qmckl_context) function qmckl_context_create() bind(C)
       use, intrinsic :: iso_c_binding
       import
     end function qmckl_context_create
  end interface
  
  interface
     integer (qmckl_exit_code) function qmckl_context_destroy(context) bind(C)
       use, intrinsic :: iso_c_binding
       import
       integer (qmckl_context), intent(in), value :: context
     end function qmckl_context_destroy
  end interface
 
end module qmckl_dgemm
