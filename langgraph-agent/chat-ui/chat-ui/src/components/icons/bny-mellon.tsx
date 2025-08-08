import React from "react";
import Image from "next/image";

export const BNYMellonLogoSVG = ({ className }: { className?: string }) => {
  return (
    <Image
      src="/logo.png"
      alt="BNY Mellon Logo"
      width={120}
      height={24}
      className={className}
    />
  );
};