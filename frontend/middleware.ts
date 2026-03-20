import { NextRequest, NextResponse } from 'next/server'

export function middleware(request: NextRequest) {
  const apiKey = request.cookies.get('cbb_api_key')?.value
  const isLoginPage = request.nextUrl.pathname === '/login'

  if (!apiKey && !isLoginPage) {
    return NextResponse.redirect(new URL('/login', request.url))
  }
  if (apiKey && isLoginPage) {
    return NextResponse.redirect(new URL('/performance', request.url))
  }
  return NextResponse.next()
}

export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico|manifest.json|icons/).*)'],
}
