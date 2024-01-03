import { card } from "../assets";
import styles, { layout } from "../style";
import Button from "./Button";

const CardDeal = () => (
  <section className={layout.section}>
    <div className={layout.sectionInfo}>
      <h2 className={styles.heading2}>
        Find a better voice Assistant
        <br className="sm:block hidden" /> in few easy steps.
      </h2>
      <p className={`${styles.paragraph} max-w-[470px] mt-5`}>
        Upgrade your meetings with our AI Meeting Voice Assistant—effortless
        efficiency and enhanced collaboration at your fingertips. Elevate your
        communication game today.
      </p>

      <a href="/">
        <Button styles={`mt-10`} />
        </a>
    </div>

    <div className={layout.sectionImg}>
      <img src={card} alt="billing" className="w-[100%] h-[100%]" />
    </div>
  </section>
);

export default CardDeal;
